# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import Any, Dict, List, Optional, Union

from .add_batch_tests import AddBatchTests
from .async_base_client import AsyncBaseClient
from .base_model import UNSET, UnsetType
from .batch_add_question_answer_pairs import BatchAddQuestionAnswerPairs
from .create_or_update_test_suite import CreateOrUpdateTestSuite
from .create_question_answer_set import CreateQuestionAnswerSet
from .create_rag_suite import CreateRagSuite
from .delete_test_suite import DeleteTestSuite
from .get_operators import GetOperators
from .get_rag_suites import GetRagSuites
from .get_test_data import GetTestData
from .get_test_suite_data import GetTestSuiteData
from .get_test_suites_with_count import GetTestSuitesWithCount
from .input_types import (
    CheckInputType,
    LocalEvalUploadInputType,
    ParameterInputType,
    QuestionAnswerPairInputType,
    TestMutationInfo,
)
from .list_question_answer_pairs import ListQuestionAnswerPairs
from .list_runs import ListRuns
from .mark_question_answer_set_as_complete import MarkQuestionAnswerSetAsComplete
from .pull_run import PullRun
from .remove_old_tests import RemoveOldTests
from .rerun_tests import RerunTests
from .run_param_info import RunParamInfo
from .run_status import RunStatus
from .start_run import StartRun
from .update_global_checks import UpdateGlobalChecks
from .update_run_status import UpdateRunStatus
from .upload_local_evaluation import UploadLocalEvaluation


def gql(q: str) -> str:
    return q


class Client(AsyncBaseClient):
    async def create_question_answer_set(
        self,
        test_suite_id: str,
        question_answer_pairs: List[QuestionAnswerPairInputType],
        parameters: Any,
        model_id: str,
        run_name: Union[Optional[str], UnsetType] = UNSET,
        **kwargs: Any
    ) -> CreateQuestionAnswerSet:
        query = gql(
            """
            mutation CreateQuestionAnswerSet($testSuiteId: String!, $questionAnswerPairs: [QuestionAnswerPairInputType!]!, $parameters: GenericScalar!, $modelId: String!, $runName: String) {
              createQuestionAnswerSet(
                testSuiteId: $testSuiteId
                questionAnswerPairs: $questionAnswerPairs
                parameters: $parameters
                modelId: $modelId
                runName: $runName
                createRun: true
              ) {
                questionAnswerSet {
                  id
                }
                runId
              }
            }
            """
        )
        variables: Dict[str, object] = {
            "testSuiteId": test_suite_id,
            "questionAnswerPairs": question_answer_pairs,
            "parameters": parameters,
            "modelId": model_id,
            "runName": run_name,
        }
        response = await self.execute(
            query=query,
            operation_name="CreateQuestionAnswerSet",
            variables=variables,
            **kwargs
        )
        data = self.get_data(response)
        return CreateQuestionAnswerSet.model_validate(data)

    async def batch_add_question_answer_pairs(
        self,
        question_answer_set_id: str,
        question_answer_pairs: List[QuestionAnswerPairInputType],
        **kwargs: Any
    ) -> BatchAddQuestionAnswerPairs:
        query = gql(
            """
            mutation BatchAddQuestionAnswerPairs($questionAnswerSetId: String!, $questionAnswerPairs: [QuestionAnswerPairInputType!]!) {
              batchAddQuestionAnswerPairs(
                questionAnswerSetId: $questionAnswerSetId
                questionAnswerPairs: $questionAnswerPairs
              ) {
                questionAnswerPairs {
                  id
                  inputUnderTest
                  llmOutput
                  fileIds
                  context
                  outputContext
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {
            "questionAnswerSetId": question_answer_set_id,
            "questionAnswerPairs": question_answer_pairs,
        }
        response = await self.execute(
            query=query,
            operation_name="BatchAddQuestionAnswerPairs",
            variables=variables,
            **kwargs
        )
        data = self.get_data(response)
        return BatchAddQuestionAnswerPairs.model_validate(data)

    async def mark_question_answer_set_as_complete(
        self, question_answer_set_id: str, **kwargs: Any
    ) -> MarkQuestionAnswerSetAsComplete:
        query = gql(
            """
            mutation MarkQuestionAnswerSetAsComplete($questionAnswerSetId: String!) {
              markQuestionAnswerSetAsComplete(questionAnswerSetId: $questionAnswerSetId) {
                questionAnswerSet {
                  id
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {"questionAnswerSetId": question_answer_set_id}
        response = await self.execute(
            query=query,
            operation_name="MarkQuestionAnswerSetAsComplete",
            variables=variables,
            **kwargs
        )
        data = self.get_data(response)
        return MarkQuestionAnswerSetAsComplete.model_validate(data)

    async def list_question_answer_pairs(
        self,
        qa_set_id: str,
        offset: Union[Optional[int], UnsetType] = UNSET,
        limit: Union[Optional[int], UnsetType] = UNSET,
        **kwargs: Any
    ) -> ListQuestionAnswerPairs:
        query = gql(
            """
            query ListQuestionAnswerPairs($qaSetId: String!, $offset: Int, $limit: Int) {
              questionAnswerPairsWithCount(
                qaSetId: $qaSetId
                filterOptions: {offset: $offset, limit: $limit}
              ) {
                questionAnswerPairs {
                  id
                  inputUnderTest
                  llmOutput
                  context
                  outputContext
                  typedMetadata {
                    inTokens
                    outTokens
                    durationSeconds
                  }
                  typedFileIds
                  localEvals {
                    id
                    score
                    feedback
                    createdAt
                  }
                  test {
                    testId
                  }
                }
                count
              }
            }
            """
        )
        variables: Dict[str, object] = {
            "qaSetId": qa_set_id,
            "offset": offset,
            "limit": limit,
        }
        response = await self.execute(
            query=query,
            operation_name="ListQuestionAnswerPairs",
            variables=variables,
            **kwargs
        )
        data = self.get_data(response)
        return ListQuestionAnswerPairs.model_validate(data)

    async def start_run(
        self,
        test_suite_id: str,
        typed_parameters: ParameterInputType,
        qa_set_id: Union[Optional[str], UnsetType] = UNSET,
        run_name: Union[Optional[str], UnsetType] = UNSET,
        run_id: Union[Optional[str], UnsetType] = UNSET,
        **kwargs: Any
    ) -> StartRun:
        query = gql(
            """
            mutation startRun($test_suite_id: String!, $typed_parameters: ParameterInputType!, $qa_set_id: String = null, $run_name: String = null, $run_id: String = null) {
              startRun(
                testSuiteId: $test_suite_id
                typedParameters: $typed_parameters
                qaSetId: $qa_set_id
                runName: $run_name
                runId: $run_id
              ) {
                runId
              }
            }
            """
        )
        variables: Dict[str, object] = {
            "test_suite_id": test_suite_id,
            "typed_parameters": typed_parameters,
            "qa_set_id": qa_set_id,
            "run_name": run_name,
            "run_id": run_id,
        }
        response = await self.execute(
            query=query, operation_name="startRun", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return StartRun.model_validate(data)

    async def update_run_status(self, run_id: str, **kwargs: Any) -> UpdateRunStatus:
        query = gql(
            """
            mutation updateRunStatus($run_id: String!) {
              updateRunStatus(runId: $run_id, status: ERROR) {
                run {
                  runId
                  status
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {"run_id": run_id}
        response = await self.execute(
            query=query, operation_name="updateRunStatus", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return UpdateRunStatus.model_validate(data)

    async def run_param_info(self, **kwargs: Any) -> RunParamInfo:
        query = gql(
            """
            query RunParamInfo {
              runParameterInfo
            }
            """
        )
        variables: Dict[str, object] = {}
        response = await self.execute(
            query=query, operation_name="RunParamInfo", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return RunParamInfo.model_validate(data)

    async def run_status(self, run_id: str, **kwargs: Any) -> RunStatus:
        query = gql(
            """
            query RunStatus($runId: String!) {
              run(runId: $runId) {
                status
              }
            }
            """
        )
        variables: Dict[str, object] = {"runId": run_id}
        response = await self.execute(
            query=query, operation_name="RunStatus", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return RunStatus.model_validate(data)

    async def pull_run(self, run_id: str, **kwargs: Any) -> PullRun:
        query = gql(
            """
            query PullRun($runId: String!) {
              run(runId: $runId) {
                qaSet {
                  id
                }
                runId
                passPercentage
                status
                textSummary
                timestamp
                completedAt
                archived
                name
                typedParameters {
                  evalModel
                  maximumThreads
                  runGoldenEval
                  runConfidenceEvaluation
                  heavyweightFactor
                  createTextSummary
                  modelUnderTest
                  temperature
                  maxOutputTokens
                  systemPrompt
                  newLineStopOption
                }
                passRate {
                  value
                  error
                }
                successRate {
                  value
                  error
                }
                testSuite {
                  id
                  title
                }
              }
              testResults(runId: $runId) {
                id
                llmOutput
                passPercentage
                passPercentageWithOptional
                resultJson
                qaPair {
                  context
                  outputContext
                }
                test {
                  testId
                  inputUnderTest
                  context
                }
                metadata
              }
            }
            """
        )
        variables: Dict[str, object] = {"runId": run_id}
        response = await self.execute(
            query=query, operation_name="PullRun", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return PullRun.model_validate(data)

    async def list_runs(
        self,
        archived: Union[Optional[bool], UnsetType] = UNSET,
        suite_id: Union[Optional[str], UnsetType] = UNSET,
        limit: Union[Optional[int], UnsetType] = UNSET,
        offset: Union[Optional[int], UnsetType] = UNSET,
        search: Union[Optional[str], UnsetType] = UNSET,
        **kwargs: Any
    ) -> ListRuns:
        query = gql(
            """
            query ListRuns($archived: Boolean, $suiteId: String, $limit: Int, $offset: Int, $search: String) {
              runsWithCount(
                filterOptions: {archived: $archived, suiteId: $suiteId, limit: $limit, offset: $offset, sortBy: STARTED_AT, search: $search}
              ) {
                runResults {
                  runId
                  passPercentage
                  passRate {
                    value
                    error
                  }
                  successRate {
                    value
                    error
                  }
                  name
                  status
                  textSummary
                  timestamp
                  completedAt
                  archived
                  typedParameters {
                    evalModel
                    maximumThreads
                    runGoldenEval
                    runConfidenceEvaluation
                    heavyweightFactor
                    createTextSummary
                    modelUnderTest
                    temperature
                    maxOutputTokens
                    systemPrompt
                    newLineStopOption
                  }
                  testSuite {
                    title
                  }
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {
            "archived": archived,
            "suiteId": suite_id,
            "limit": limit,
            "offset": offset,
            "search": search,
        }
        response = await self.execute(
            query=query, operation_name="ListRuns", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return ListRuns.model_validate(data)

    async def create_or_update_test_suite(
        self, test_suite_id: str, title: str, description: str, **kwargs: Any
    ) -> CreateOrUpdateTestSuite:
        query = gql(
            """
            mutation createOrUpdateTestSuite($testSuiteId: String!, $title: String!, $description: String!) {
              updateTestSuite(
                testSuiteId: $testSuiteId
                title: $title
                description: $description
              ) {
                testSuite {
                  description
                  id
                  org
                  title
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {
            "testSuiteId": test_suite_id,
            "title": title,
            "description": description,
        }
        response = await self.execute(
            query=query,
            operation_name="createOrUpdateTestSuite",
            variables=variables,
            **kwargs
        )
        data = self.get_data(response)
        return CreateOrUpdateTestSuite.model_validate(data)

    async def add_batch_tests(
        self, tests: List[TestMutationInfo], create_only: bool, **kwargs: Any
    ) -> AddBatchTests:
        query = gql(
            """
            mutation addBatchTests($tests: [TestMutationInfo!]!, $createOnly: Boolean!) {
              batchUpdateTest(tests: $tests, createOnly: $createOnly) {
                tests {
                  checks
                  testId
                  crossVersionId
                  fileIds
                  inputUnderTest
                  tags
                  context
                  goldenOutput
                  testSuite {
                    id
                  }
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {"tests": tests, "createOnly": create_only}
        response = await self.execute(
            query=query, operation_name="addBatchTests", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return AddBatchTests.model_validate(data)

    async def delete_test_suite(
        self, test_suite_id: str, **kwargs: Any
    ) -> DeleteTestSuite:
        query = gql(
            """
            mutation deleteTestSuite($testSuiteId: String!) {
              deleteSuite(suiteId: $testSuiteId) {
                success
              }
            }
            """
        )
        variables: Dict[str, object] = {"testSuiteId": test_suite_id}
        response = await self.execute(
            query=query, operation_name="deleteTestSuite", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return DeleteTestSuite.model_validate(data)

    async def update_global_checks(
        self, test_suite_id: str, checks: List[CheckInputType], **kwargs: Any
    ) -> UpdateGlobalChecks:
        query = gql(
            """
            mutation updateGlobalChecks($testSuiteId: String!, $checks: [CheckInputType!]!) {
              updateGlobalChecks(testSuiteId: $testSuiteId, checks: $checks) {
                success
              }
            }
            """
        )
        variables: Dict[str, object] = {"testSuiteId": test_suite_id, "checks": checks}
        response = await self.execute(
            query=query,
            operation_name="updateGlobalChecks",
            variables=variables,
            **kwargs
        )
        data = self.get_data(response)
        return UpdateGlobalChecks.model_validate(data)

    async def remove_old_tests(
        self, test_suite_id: str, in_use_tests: List[str], **kwargs: Any
    ) -> RemoveOldTests:
        query = gql(
            """
            mutation removeOldTests($testSuiteId: String!, $inUseTests: [String!]!) {
              removeUnusedTests(testSuiteId: $testSuiteId, inUseTests: $inUseTests) {
                success
              }
            }
            """
        )
        variables: Dict[str, object] = {
            "testSuiteId": test_suite_id,
            "inUseTests": in_use_tests,
        }
        response = await self.execute(
            query=query, operation_name="removeOldTests", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return RemoveOldTests.model_validate(data)

    async def create_rag_suite(
        self, query: str, file_path: str, **kwargs: Any
    ) -> CreateRagSuite:
        _query = gql(
            """
            mutation createRagSuite($query: String!, $filePath: String!) {
              updateRagSuite(ragSuiteId: "0", query: $query, filePath: $filePath) {
                ragSuite {
                  id
                  query
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {"query": query, "filePath": file_path}
        response = await self.execute(
            query=_query, operation_name="createRagSuite", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return CreateRagSuite.model_validate(data)

    async def rerun_tests(self, run_id: str, **kwargs: Any) -> RerunTests:
        query = gql(
            """
            mutation RerunTests($runId: String!) {
              rerunFailingTests(runId: $runId) {
                success
              }
            }
            """
        )
        variables: Dict[str, object] = {"runId": run_id}
        response = await self.execute(
            query=query, operation_name="RerunTests", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return RerunTests.model_validate(data)

    async def upload_local_evaluation(
        self,
        question_answer_set_id: str,
        local_evals: List[LocalEvalUploadInputType],
        **kwargs: Any
    ) -> UploadLocalEvaluation:
        query = gql(
            """
            mutation uploadLocalEvaluation($questionAnswerSetId: String!, $localEvals: [LocalEvalUploadInputType!]!) {
              uploadLocalEvaluation(
                questionAnswerSetId: $questionAnswerSetId
                localEvals: $localEvals
              ) {
                uploads {
                  id
                  score
                  feedback
                  createdAt
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {
            "questionAnswerSetId": question_answer_set_id,
            "localEvals": local_evals,
        }
        response = await self.execute(
            query=query,
            operation_name="uploadLocalEvaluation",
            variables=variables,
            **kwargs
        )
        data = self.get_data(response)
        return UploadLocalEvaluation.model_validate(data)

    async def get_test_suite_data(
        self, suite_id: str, **kwargs: Any
    ) -> GetTestSuiteData:
        query = gql(
            """
            query getTestSuiteData($suiteId: String!) {
              testSuites(testSuiteId: $suiteId) {
                description
                id
                org
                title
                created
                globalChecks
              }
            }
            """
        )
        variables: Dict[str, object] = {"suiteId": suite_id}
        response = await self.execute(
            query=query,
            operation_name="getTestSuiteData",
            variables=variables,
            **kwargs
        )
        data = self.get_data(response)
        return GetTestSuiteData.model_validate(data)

    async def get_test_data(self, suite_id: str, **kwargs: Any) -> GetTestData:
        query = gql(
            """
            query getTestData($suiteId: String!) {
              tests(testSuiteId: $suiteId) {
                checks
                testId
                crossVersionId
                fileIds
                inputUnderTest
                tags
                context
                goldenOutput
                testSuite {
                  id
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {"suiteId": suite_id}
        response = await self.execute(
            query=query, operation_name="getTestData", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return GetTestData.model_validate(data)

    async def get_test_suites_with_count(
        self,
        offset: Union[Optional[int], UnsetType] = UNSET,
        limit: Union[Optional[int], UnsetType] = UNSET,
        search: Union[Optional[str], UnsetType] = UNSET,
        **kwargs: Any
    ) -> GetTestSuitesWithCount:
        query = gql(
            """
            query getTestSuitesWithCount($offset: Int, $limit: Int, $search: String) {
              testSuitesWithCount(
                filterOptions: {offset: $offset, limit: $limit, search: $search}
              ) {
                testSuites {
                  id
                  title
                  description
                  created
                  creator
                  lastModifiedBy
                  lastModifiedAt
                  folder {
                    id
                    name
                  }
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {
            "offset": offset,
            "limit": limit,
            "search": search,
        }
        response = await self.execute(
            query=query,
            operation_name="getTestSuitesWithCount",
            variables=variables,
            **kwargs
        )
        data = self.get_data(response)
        return GetTestSuitesWithCount.model_validate(data)

    async def get_operators(self, **kwargs: Any) -> GetOperators:
        query = gql(
            """
            query getOperators {
              operators {
                nameInDoc
                isUnary
              }
            }
            """
        )
        variables: Dict[str, object] = {}
        response = await self.execute(
            query=query, operation_name="getOperators", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return GetOperators.model_validate(data)

    async def get_rag_suites(self, **kwargs: Any) -> GetRagSuites:
        query = gql(
            """
            query getRagSuites {
              ragSuites {
                id
                org
                path
                query
              }
            }
            """
        )
        variables: Dict[str, object] = {}
        response = await self.execute(
            query=query, operation_name="getRagSuites", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return GetRagSuites.model_validate(data)
