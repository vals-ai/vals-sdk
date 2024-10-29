# Generated by ariadne-codegen
# Source: http://localhost:8000/graphql/

from enum import Enum


class AppQuestionAnswerSetCreationMethodChoices(str, Enum):
    LIVE_QUERY = "LIVE_QUERY"
    SDK = "SDK"
    UPLOAD = "UPLOAD"


class AppQuestionAnswerSetStatusChoices(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class ExampleType(str, Enum):
    positive = "positive"
    negative = "negative"


class TestSuiteSortField(str, Enum):
    TITLE = "TITLE"
    CREATED = "CREATED"
    LAST_MODIFIED_AT = "LAST_MODIFIED_AT"


class SortOrder(str, Enum):
    ASC = "ASC"
    DESC = "DESC"
