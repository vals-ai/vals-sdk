"""
Tests for rerun_checks functionality in SDK and CLI.
requires VALS_ENV = 'DEV' w VALS_API_KEY configured.
"""

import os
import pytest

from vals.sdk.run import Run


class TestRerunChecks:
    """Test the rerun_all_checks functionality."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="This test is temporarily disabled")
    async def test_rerun_all_checks_integration(self):
        """Integration test for rerun_all_checks with specific run ID."""
        # Assert VALS_ENV is set to DEV
        vals_env = os.getenv("VALS_ENV")
        assert vals_env == "DEV", f"VALS_ENV must be set to 'DEV', got: {vals_env}"

        # Assert VALS_API_KEY is set
        vals_api_key = os.getenv("VALS_API_KEY")
        assert vals_api_key is not None, "VALS_API_KEY environment variable must be set"

        # Use the specific run ID provided
        run_id = "57d78d38-a19c-4f52-95f2-2eb4b4615b76"

        try:
            # Get the original run
            original_run = await Run.from_id(run_id)

            # Test that rerun_all_checks doesn't error
            new_run = await original_run.rerun_all_checks()

            # Verify we got a new run back
            assert new_run is not None
            assert isinstance(new_run, Run)
            assert new_run.id != original_run.id  # Should be a different run

            # Test that the resulting run has its single check pass
            # Note: This assumes the run completes quickly. In practice, you might need to wait
            await new_run.wait_for_run_completion()
            await new_run.pull()  # Refresh data from server

            # Check that all test results pass
            assert len(new_run.test_results) > 0, "Run should have test results"

            # Verify at least one check passes (auto_eval = 1.0 means pass)
            passing_checks = sum(
                1
                for test_result in new_run.test_results
                if any(check.auto_eval == 1.0 for check in test_result.check_results)
            )
            assert passing_checks > 0, "At least one check should pass"

        except Exception as e:
            pytest.fail(f"rerun_all_checks should not error, but got: {e}")


class TestRerunChecksEdgeCases:
    """Test edge cases for rerun_all_checks functionality."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="This test is temporarily disabled")
    async def test_rerun_all_checks_with_invalid_run_id(self):
        """Test rerun_all_checks with invalid run ID."""
        # Assert environment variables are set
        assert os.getenv("VALS_ENV") == "DEV", "VALS_ENV must be set to 'DEV'"
        assert os.getenv("VALS_API_KEY") is not None, "VALS_API_KEY must be set"

        with pytest.raises(Exception):  # Should raise some form of error
            invalid_run = await Run.from_id("invalid-run-id-12345")
            await invalid_run.rerun_all_checks()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="This test is temporarily disabled")
    async def test_rerun_all_checks_preserves_original_run(self):
        """Test that rerun_all_checks doesn't modify the original run."""
        # Assert environment variables are set
        assert os.getenv("VALS_ENV") == "BENCH", "VALS_ENV must be set to 'BENCH'"
        assert os.getenv("VALS_API_KEY") is not None, "VALS_API_KEY must be set"

        # run_id = "b5016745-3911-4ab1-b708-cd38ddbd5304" # small, DEV
        run_id = "e80d9cb2-d5de-4901-9c7c-4a0914e54149"  # large, BENCH

        try:
            original_run = await Run.from_id(run_id)
            original_status = original_run.status
            original_timestamp = original_run.timestamp

            new_run = await original_run.rerun_all_checks()

            # Verify original run is unchanged
            assert original_run.status == original_status
            assert original_run.timestamp == original_timestamp
            assert original_run.id != new_run.id

        except Exception as e:
            pytest.fail(f"Test failed with error: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
