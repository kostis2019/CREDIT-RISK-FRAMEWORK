# SETTINGS: unsecured loans dataset

# INITIAL COLUMNS

COLUMN_CASE_ID = "Loan_ID"
COLUMN_CLIENT_ID = "Client_ID"
COLUMN_DATE = "LoanDate"
COLUMN_TARGET = "default12"
COLUMN_TARGET_DATE = "DefaultDate"
COLUMN_EAD = "Amount"

# DERIVED COLUMNS

COLUMN_YEAR = "Year"

# DISPLAY LABELS

LABEL_YEAR = "Loan Origination Year"

# EXCLUDE COLUMNS FROM PD TRAINING

PD_EXCLUDE_FEATURES = [
    COLUMN_DATE,
    COLUMN_TARGET_DATE,
    COLUMN_YEAR,
    "Country",
    "CreditScoreEsMicroL",
    "NrOfDependants",
    "LoanApplicationStartedDate",
    "FirstPaymentDate",
]

PD_FIRST_ITERATION_FEATURES = [
    "NewCreditCustomer",
    "LoanApplicationStartedDate",
    "LoanDate",
    "FirstPaymentDate",
    "ApplicationSignedHour",
    "ApplicationSignedWeekday",
    "VerificationType",
    "LanguageCode",
    "Age",
    "Gender",
    "AppliedAmount",
    "Amount",
    #"Interest",
    "LoanDuration",
    "MonthlyPayment",
    "UseOfLoan",
    "Education",
    "MaritalStatus",
    #"NrOfDependants",
    "EmploymentStatus",
    "EmploymentDurationCurrentEmployer",
    "WorkExperience",
    "OccupationArea",
    "HomeOwnershipType",
    "IncomeFromPrincipalEmployer",
    "IncomeFromPension",
    "IncomeFromFamilyAllowance",
    "IncomeFromSocialWelfare",
    "IncomeFromLeavePay",
    "IncomeFromChildSupport",
    "IncomeOther",
    "IncomeTotal",
    "ExistingLiabilities",
    "LiabilitiesTotal",
    "RefinanceLiabilities",
    "DebtToIncome",
    "FreeCash",
    "MonthlyPaymentDay",
    "DefaultDate",
    #"CreditScoreEsMicroL",
    "CreditScoreEeMini",
    "NoOfPreviousLoansBeforeLoan",
    "AmountOfPreviousLoansBeforeLoan",
    "PreviousRepaymentsBeforeLoan",
    "PreviousEarlyRepaymentsBefoleLoan",
    "PreviousEarlyRepaymentsCountBeforeLoan",
    "Country",
]