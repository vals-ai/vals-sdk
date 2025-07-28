"""Tests object mappings between backend and SDK."""

from datetime import datetime

from vals.graphql_client import (
    CheckModifierOutputTypeFields,
    CheckModifierOutputTypeFieldsConditional,
    CheckModifierOutputTypeFieldsExamples,
    CheckOutputTypeFields,
    CheckOutputTypeFieldsModifiers,
    GetTestSuitesWithCountTestSuitesWithCountTestSuites,
    GetTestSuitesWithCountTestSuitesWithCountTestSuitesFolder,
    ListRunsRunsWithCountRunResults,
    ListRunsRunsWithCountRunResultsParameters,
    ListRunsRunsWithCountRunResultsPassRate,
    ListRunsRunsWithCountRunResultsSuccessRate,
    ListRunsRunsWithCountRunResultsTestSuite,
    PullTestResultsWithCountTestResultsWithCountTestResults,
    PullTestResultsWithCountTestResultsWithCountTestResultsMetadata,
    PullTestResultsWithCountTestResultsWithCountTestResultsQaPair,
    PullTestResultsWithCountTestResultsWithCountTestResultsResultJson,
    PullTestResultsWithCountTestResultsWithCountTestResultsTest,
    ResultJsonElementTypeFieldsModifiers,
    TestFragmentChecks,
    TestFragmentChecksModifiers,
    TestFragmentTestSuite,
)
from vals.graphql_client.enums import RunStatus
from vals.sdk.types import (
    Check,
    CheckModifiers,
    RunMetadata,
    TestResult,
    TestSuiteMetadata,
)


class TestObjectMappings:
    def test_test_suite_metadata(self):
        """Test mapping from TestSuiteType to TestSuiteMetadata"""
        gql_suite = GetTestSuitesWithCountTestSuitesWithCountTestSuites(
            id="suite123",
            title="My Suite",
            description="A test suite",
            created=datetime(2024, 1, 1),
            creator="alice",
            lastModifiedBy="bob",
            lastModifiedAt=datetime(2024, 1, 2),
            folder=GetTestSuitesWithCountTestSuitesWithCountTestSuitesFolder(
                id="folder123",
                name="My Folder",
            ),
        )

        result = TestSuiteMetadata.from_graphql(gql_suite)

        assert result.id == "suite123"
        assert result.title == "My Suite"
        assert result.description == "A test suite"
        assert result.created == datetime(2024, 1, 1)
        assert result.creator == "alice"
        assert result.last_modified_by == "bob"
        assert result.last_modified_at == datetime(2024, 1, 2)
        assert result.folder_id == "folder123"
        assert result.folder_name == "My Folder"

    def test_check_modifiers(self):
        gql_example_pos = CheckModifierOutputTypeFieldsExamples(
            type="positive",
            text="Example text",
        )

        gql_example_neg = CheckModifierOutputTypeFieldsExamples(
            type="negative",
            text="Example text",
        )

        gql_conditional = CheckModifierOutputTypeFieldsConditional(
            operator="Operator", criteria="Criteria"
        )

        gql_check_modifiers = CheckModifierOutputTypeFields(
            optional=True,
            severity=0.9,
            displayMetrics=True,
            examples=[gql_example_pos, gql_example_neg],
            extractor="keyword",
            conditional=gql_conditional,
            category="consistency",
        )

        result = CheckModifiers.from_graphql(gql_check_modifiers.model_dump())  # pyright: ignore[reportUnknownMemberType]

        assert result.optional is True
        assert result.severity == 0.9
        assert result.display_metrics is True
        assert result.extractor == "keyword"
        assert result.category == "consistency"

        assert len(result.examples) == 2
        assert result.examples[0].type == "positive"
        assert result.examples[0].text == "Example text"
        assert result.examples[1].type == "negative"
        assert result.examples[1].text == "Example text"

        assert result.conditional is not None
        assert result.conditional.operator == "Operator"
        assert result.conditional.criteria == "Criteria"

    def test_check(self):
        gql_example_pos = CheckModifierOutputTypeFieldsExamples(
            type="positive",
            text="Example text",
        )

        gql_example_neg = CheckModifierOutputTypeFieldsExamples(
            type="negative",
            text="Example text",
        )

        gql_conditional = CheckModifierOutputTypeFieldsConditional(
            operator="Operator", criteria="Criteria"
        )

        gql_check_modifiers = CheckOutputTypeFieldsModifiers(
            optional=True,
            severity=0.9,
            displayMetrics=True,
            examples=[gql_example_pos, gql_example_neg],
            extractor="keyword",
            conditional=gql_conditional,
            category="consistency",
        )

        gql_check = CheckOutputTypeFields(
            operator="Check",
            criteria="Criteria",
            modifiers=gql_check_modifiers,
        )

        result = Check.from_graphql(gql_check.model_dump())  # pyright: ignore[reportUnknownMemberType]
        assert result.operator == "Check"
        assert result.criteria == "Criteria"

        modifiers = result.modifiers
        assert modifiers.optional is True
        assert modifiers.severity == 0.9
        assert modifiers.display_metrics is True
        assert modifiers.extractor == "keyword"
        assert modifiers.category == "consistency"

        assert isinstance(modifiers.examples, list)
        assert len(modifiers.examples) == 2

        assert modifiers.examples[0].type == "positive"
        assert modifiers.examples[0].text == "Example text"
        assert modifiers.examples[1].type == "negative"
        assert modifiers.examples[1].text == "Example text"

        cond = modifiers.conditional
        assert cond is not None
        assert cond.operator == "Operator"
        assert cond.criteria == "Criteria"

    def test_run_metadata(self):
        mock_pass_rate = ListRunsRunsWithCountRunResultsPassRate(
            value=0.85,
            error=0.5,
        )

        mock_success_rate = ListRunsRunsWithCountRunResultsSuccessRate(
            value=0.9,
            error=0,
        )

        mock_parameters = ListRunsRunsWithCountRunResultsParameters(
            evalModel="gpt-4",
            maximumThreads=8,
            runConfidenceEvaluation=False,
            heavyweightFactor=2,
            createTextSummary=True,
            modelUnderTest="gpt-4",
            temperature=0.7,
            maxOutputTokens=512,
            systemPrompt="You are a helpful assistant.",
        )

        mock_test_suite = ListRunsRunsWithCountRunResultsTestSuite(
            title="Test Suite 1",
        )

        gql_run_metadata = ListRunsRunsWithCountRunResults(
            runId="run123",
            passPercentage=95.5,
            passRate=mock_pass_rate,
            successRate=mock_success_rate,
            name="Run 1",
            status=RunStatus.SUCCESS,
            textSummary="All tests passed.",
            timestamp=datetime.now(),
            completedAt=datetime.now(),
            archived=False,
            parameters=mock_parameters,
            testSuite=mock_test_suite,
        )

        result = RunMetadata.from_graphql(gql_run_metadata)

        assert result.id == "run123"
        assert result.pass_percentage == 95.5

        assert result.pass_rate is not None
        assert result.pass_rate == 0.85
        assert result.pass_rate_error == 0.5

        assert result.success_rate is not None
        assert result.success_rate == 0.9
        assert result.success_rate_error == 0

        assert result.name == "Run 1"
        assert result.status == RunStatus.SUCCESS
        assert result.text_summary == "All tests passed."

        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.completed_at, datetime)
        assert result.archived is False
        assert result.model == "gpt-4"

        params = result.parameters
        assert params.eval_model == "gpt-4"
        # assert params.maximum_threads == 8
        assert params.run_confidence_evaluation is False
        assert params.heavyweight_factor == 2
        assert params.create_text_summary is True
        assert params.temperature == 0.7
        assert params.max_output_tokens == 512
        assert params.system_prompt == "You are a helpful assistant."

        assert result.test_suite_title == "Test Suite 1"

    def test_test_result(self):
        mock_modifiers = ResultJsonElementTypeFieldsModifiers(
            optional=True,
            severity=0.8,
            extractor="keyword",
            displayMetrics=True,
            examples=[],
            conditional=None,
            category="formatting",
        )

        mock_result_json = (
            PullTestResultsWithCountTestResultsWithCountTestResultsResultJson(
                operator="equals",
                criteria="Output must contain keyword",
                modifiers=mock_modifiers,
                autoEval=0.95,
                feedback="Passed automatically",
                evalCont=0.85,
                isGlobal=False,
                severity=0.75,
            )
        )

        mock_qa_pair = PullTestResultsWithCountTestResultsWithCountTestResultsQaPair(
            context={"context": "my context"},
            outputContext={"context": "my output context"},
            errorMessage="Error message",
        )

        mock_test = PullTestResultsWithCountTestResultsWithCountTestResultsTest(
            id="test123",
            inputUnderTest="input under test",
            context={"context": "my context"},
            tags=["tag1", "tag2"],
            crossVersionId="id123",
            goldenOutput="golden output",
            checks=[
                TestFragmentChecks(
                    operator="equals",
                    criteria="Output must contain",
                    modifiers=TestFragmentChecksModifiers(
                        **mock_modifiers.model_dump()
                    ),
                )
            ],
            fileIds=[],
            testSuite=TestFragmentTestSuite(id="testsuite123"),
        )

        mock_metadata = PullTestResultsWithCountTestResultsWithCountTestResultsMetadata(
            inTokens=123,
            outTokens=123,
            durationSeconds=123.0,
        )

        mock_test_result = PullTestResultsWithCountTestResultsWithCountTestResults(
            id="result123",
            llmOutput="Paris is the capital of France.",
            passPercentage=100.0,
            passPercentageWithWeight=95.0,
            resultJson=[mock_result_json],
            qaPair=mock_qa_pair,
            test=mock_test,
            metadata=mock_metadata,
        )

        result = TestResult.from_graphql(mock_test_result)
        # assert result._id == "result123"
        assert result.input_under_test == "input under test"
        assert result.llm_output == "Paris is the capital of France."
        assert result.pass_percentage == 100.0
        assert result.pass_percentage_with_weight == 95.0
        assert result.error_message == "Error message"

        assert result.context == {"context": "my context"}
        assert result.output_context == {"context": "my output context"}

        assert result.test.input_under_test == "input under test"
        assert result.test.context == {"context": "my context"}
        assert result.test.id == "id123"
        # assert result.test.golden_output == "golden output"
        assert result.test.tags == ["tag1", "tag2"]
        # assert result.test._file_ids == []
        # assert result.test._test_suite_id == "testsuite123"

        assert len(result.check_results) == 1
        check_result = result.check_results[0]
        assert check_result.operator == "equals"
        assert check_result.criteria == "Output must contain keyword"
        assert check_result.is_global is False
        assert check_result.auto_eval == 0.95
        assert check_result.feedback == "Passed automatically"
        assert (
            check_result.confidence.value == 0.5
        )  # because 0.85 is not in [0, 0.5, 1]

        mod = check_result.modifiers
        assert mod.optional is True
        assert mod.severity == 0.8
        assert mod.extractor == "keyword"
        assert mod.display_metrics is True
        assert mod.category == "formatting"
        assert mod.examples == []  # no examples
        assert mod.conditional is None

        assert result.metadata is not None
        assert result.metadata.in_tokens == 123
        assert result.metadata.out_tokens == 123
        assert result.metadata.duration_seconds == 123.0
