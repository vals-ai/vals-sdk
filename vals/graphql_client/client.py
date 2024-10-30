# Generated by ariadne-codegen
# Source: vals/graphql/

from typing import Any, Dict, List, Optional, Union

from .add_batch_tests import AddBatchTests
from .async_base_client import AsyncBaseClient
from .base_model import UNSET, UnsetType
from .create_or_update_test_suite import CreateOrUpdateTestSuite
from .delete_test_suite import DeleteTestSuite
from .get_test_data import GetTestData
from .get_test_suite_data import GetTestSuiteData
from .get_test_suites import GetTestSuites
from .input_types import TestMutationInfo
from .list_runs import ListRuns
from .pull_run import PullRun
from .remove_old_tests import RemoveOldTests
from .run_param_info import RunParamInfo
from .run_status import RunStatus
from .start_run import StartRun
from .update_global_checks import UpdateGlobalChecks


def gql(q: str) -> str:
    return q


class Client(AsyncBaseClient):
    async def start_run(
        self,
        test_suite_id: str,
        parameters: Any,
        qa_set_id: Union[Optional[str], UnsetType] = UNSET,
        **kwargs: Any
    ) -> StartRun:
        query = gql(
            """
            mutation startRun($test_suite_id: String!, $parameters: GenericScalar!, $qa_set_id: String = null) {
              startRun(
                testSuiteId: $test_suite_id
                parameters: $parameters
                qaSetId: $qa_set_id
              ) {
                runId
              }
            }
            """
        )
        variables: Dict[str, object] = {
            "test_suite_id": test_suite_id,
            "parameters": parameters,
            "qa_set_id": qa_set_id,
        }
        response = await self.execute(
            query=query, operation_name="startRun", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return StartRun.model_validate(data)

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
                runId
                passPercentage
                status
                runId
                textSummary
                timestamp
                completedAt
                archived
                parameters
                testSuite {
                  title
                }
              }
              testResults(runId: $runId) {
                id
                llmOutput
                passPercentage
                passPercentageWithOptional
                resultJson
                humanEval
                humanFeedback
                test {
                  testId
                  inputUnderTest
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
        **kwargs: Any
    ) -> ListRuns:
        query = gql(
            """
            query ListRuns($archived: Boolean, $suiteId: String) {
              runs(archived: $archived, suiteId: $suiteId) {
                runId
                passPercentage
                status
                runId
                textSummary
                timestamp
                completedAt
                archived
                parameters
                testSuite {
                  title
                }
              }
            }
            """
        )
        variables: Dict[str, object] = {"archived": archived, "suiteId": suite_id}
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
                  testId
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
        self, test_suite_id: str, checks: str, **kwargs: Any
    ) -> UpdateGlobalChecks:
        query = gql(
            """
            mutation updateGlobalChecks($testSuiteId: String!, $checks: String!) {
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

    async def get_test_suites(self, **kwargs: Any) -> GetTestSuites:
        query = gql(
            """
            query getTestSuites {
              testSuites {
                description
                id
                org
                title
                created
                creator
              }
            }
            """
        )
        variables: Dict[str, object] = {}
        response = await self.execute(
            query=query, operation_name="getTestSuites", variables=variables, **kwargs
        )
        data = self.get_data(response)
        return GetTestSuites.model_validate(data)
