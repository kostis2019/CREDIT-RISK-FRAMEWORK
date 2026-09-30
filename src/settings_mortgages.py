# SETTINGS: mortgages dataset

# INITIAL COLUMNS

COLUMN_CASE_ID = "Loan_ID"
COLUMN_CLIENT_ID = "Client_ID"
COLUMN_DATE = "DateOfObservation"
COLUMN_TARGET = "DefaultFlag"
COLUMN_TARGET_DATE = "-"
COLUMN_EAD = "ExposureAmount"

# DERIVED COLUMNS

COLUMN_YEAR = "Year"

# DISPLAY LABELS

LABEL_YEAR = "Observation Year"

# FEATURES: STRUCTURAL AND BEHAVIOURAL

FEATURES_STRUCTURAL = [
    "PropertyValue",
    "PropertySize",
    "ExposureLoanToValue",
    "TotalCustomerLoanToValue",
    "PropertyType",
    "ExposureAmount",
    "RemainingPaymentsRatio",
    "TimeToMaturity",
    "MaturityRatio",
    "InterestRate",
    "MonthsOnBook",
    "NumberOfExposures",
    ]

FEATURES_BEHAVIOURAL = [
    "DelinquencyFlag",
    "DelinquencyLast3Mon",
    "DelinquencyLast12Mon",
    "30PlusDelinquencyLast3Mon",
    "30PlusDelinquencyLast12Mon",
    "60PlusDelinquencyLast3Mon",
    "60PlusDelinquencyLast12Mon",
    "0_30DelinquencyLast3Mon",
    "0_30DelinquencyLast12Mon",
    "30_60DelinquencyLast3Mon",
    "30_60DelinquencyLast12Mon",
    "60_90DelinquencyLast3Mon",
    "60_90DelinquencyLast12Mon",
    "DaysInDelinquency",
    "ExposureDefaultFlagCount",
    "ClientDefaultFlagCount",
    ]

EXCLUDE_FEATURES = [
    COLUMN_DATE,
    COLUMN_TARGET_DATE,
    COLUMN_YEAR,
]