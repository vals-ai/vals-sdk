#!/usr/bin/env python3
"""
Test script for the new rerun_all_checks method
"""

import asyncio

async def test_rerun_all_checks():
    """Test the rerun_all_checks method on a specific run"""
    
    # Configure API credentials
    import os
    os.environ['VALS_ENV'] = 'LOCAL'
    os.environ['VALS_SERVER_ENV'] = 'LOCAL'
    os.environ['VALS_API_KEY'] = 'K2yvpMDDnHSsqU67xHr8obJjeGjvtPqdnwPnxQHFBA8C1hmilcmItUXXU6b4MzoIlUTz0Mn'
    
    # Use the run ID provided by the user
    run_id = '50ee3435-d599-402b-ba9a-4f890b7dd534'

    from vals.sdk.run import Run
    
    print(f"Testing rerun_all_checks on run ID: {run_id}")
    
    try:
        # Load the run
        print("Loading run...")
        run = await Run.from_id(run_id)
        
        print(f"Run name: {run.name}")
        print(f"Current status: {run.status}")
        print(f"Test results count: {len(run.test_results)}")
        print(f"Test suite: {run.test_suite_title}")
        
        # Test the new method
        print("\nCalling rerun_all_checks()...")
        await run.rerun_all_checks()
        
        print("✅ rerun_all_checks() completed successfully!")
        
        # Check the status after
        await run.pull()
        print(f"Status after rerun: {run.status}")
        
    except Exception as e:
        print(f"❌ Error testing rerun_all_checks: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_rerun_all_checks())