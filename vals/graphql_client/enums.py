# Generated by ariadne-codegen
# Source: http://localhost:8000/graphql/

from enum import Enum


class AppQuestionAnswerSetCreationMethodChoices(str, Enum):
    LIVE_QUERY = "LIVE_QUERY"
    SDK = "SDK"
    UPLOAD = "UPLOAD"


class RunStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    RERUNNING = "RERUNNING"


class TestResultReviewStatusEnum(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"


class RunReviewStatusEnum(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"
    CANCELLED = "CANCELLED"


class AppPairwiseRunReviewWinningRunChoices(str, Enum):
    A = "A"
    B = "B"
    TIE = "TIE"


class AppPairwiseTestResultReviewWinningRunChoices(str, Enum):
    A = "A"
    B = "B"
    TIE = "TIE"


class TierEnum(str, Enum):
    FREE = "FREE"
    STARTUP = "STARTUP"
    ENTERPRISE = "ENTERPRISE"


class TemplateType(str, Enum):
    CATEGORICAL = "CATEGORICAL"
    NUMERICAL = "NUMERICAL"
    FREE_TEXT = "FREE_TEXT"


class RunReviewTableSortField(str, Enum):
    CREATED_AT = "CREATED_AT"
    COMPLETED_TIME = "COMPLETED_TIME"
    STATUS = "STATUS"


class SortOrder(str, Enum):
    ASC = "ASC"
    DESC = "DESC"


class TestResultReviewSortField(str, Enum):
    STARTED_AT = "STARTED_AT"
    STATUS = "STATUS"
    COMPLETED_AT = "COMPLETED_AT"


class RunResultSortField(str, Enum):
    NAME = "NAME"
    STARTED_AT = "STARTED_AT"
    COMPLETED_AT = "COMPLETED_AT"
    PASS_PERCENTAGE = "PASS_PERCENTAGE"


class TestResultCheckErrorEnum(str, Enum):
    ALL = "ALL"
    SOME_CHECKS_FAILED = "SOME_CHECKS_FAILED"
    ALL_CHECKS_FAILED = "ALL_CHECKS_FAILED"
    ALL_CHECKS_PASSED = "ALL_CHECKS_PASSED"
    TESTS_WITH_OUTPUT_ERRORS = "TESTS_WITH_OUTPUT_ERRORS"


class TestSuiteSortField(str, Enum):
    TITLE = "TITLE"
    CREATED = "CREATED"
    LAST_MODIFIED_AT = "LAST_MODIFIED_AT"


class WinningRunEnum(str, Enum):
    A = "A"
    B = "B"
    TIE = "TIE"
