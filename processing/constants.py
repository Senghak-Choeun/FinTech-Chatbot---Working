INTENT_RESPONSE_TEMPLATE = {
    "balance_inquiry": "You can check your latest balance in the app dashboard under Accounts.",
    "fund_transfer": "To transfer funds, select recipient, amount, and confirm with secure verification.",
    "fraud_report": "Please report the suspicious transaction immediately and lock your card for safety.",
    "loan_info": "Loan options and eligibility can be reviewed in the Loans section.",
    "kyc_verification": "KYC verification requires identity details and document confirmation.",
    "account_opening": "You can open an account online by completing the onboarding steps.",
    "bill_payment": "Bill payments are available in Payments and Bills, including recurring setup.",
    "card_issue": "For card issues, please open Card Support to freeze, replace, or troubleshoot.",
    "transaction_status": "You can track the latest transaction status in your transaction history.",
    "fee_inquiry": "Fees depend on account plan and transaction type. Check fee preview before confirmation.",
    "cash_withdrawal": "You can withdraw cash using card-enabled ATM services supported by your account.",
    "wallet_topup": "Wallet top up is available through linked bank card or bank transfer options.",
}

BANKING77_TO_CORE = {
    "balance_inquiry": [
        "beneficiary_not_verified",
        "beneficiary_not_allowed",
        "balance_not_updated_after_cheque_or_bank_transfer",
    ],
    "fund_transfer": [
        "beneficiary_not_allowed",
        "beneficiary_not_verified",
        "wrong_bank_transfer",
        "pending_bank_transfer",
        "transfer_not_received_by_recipient",
        "declined_transfer",
    ],
    "fraud_report": [
        "cash_withdrawal_not_recognised",
        "card_payment_not_recognised",
    ],
    "card_issue": [
        "card_not_working",
        "card_not_received",
        "card_arrival",
        "cash_withdrawal_card",
        "beneficiary_not_allowed",
    ],
    "transaction_status": [
        "pending_bank_transfer",
        "declined_transfer",
        "transfer_not_received_by_recipient",
        "card_payment_reverted",
        "cash_withdrawal",
        "beneficiary_not_allowed",
    ],
    "fee_inquiry": [
        "cash_withdrawal_charge",
        "card_payment_fee_charged",
        "exchange_fee",
    ],
    "cash_withdrawal": [
        "cash_withdrawal",
        "cash_withdrawal_not_recognised",
        "cash_withdrawal_charge",
    ],
    "wallet_topup": [
        "top_up_by_bank_card",
        "top_up_by_bank_card_pending",
        "top_up_by_bank_card_declined",
    ],
}

SYNTHETIC_REGIONAL_PATTERNS = {
    "balance_inquiry": [
        "check my aba balance now",
        "show wing wallet balance",
        "how much money left in my account",
        "my truemoney balance please",
    ],
    "fund_transfer": [
        "send 20 dollars to aba account",
        "transfer money from wing to bank",
        "how to transfer to friend account",
        "bank transfer to local account",
    ],
    "fraud_report": [
        "unknown charge on my aba card",
        "i did not make this transfer",
        "someone used my card",
        "report fraud transaction now",
    ],
    "loan_info": [
        "how to apply personal loan",
        "what is loan interest rate",
        "how long loan approval takes",
        "am i eligible for quick loan",
    ],
    "kyc_verification": [
        "how to verify my identity",
        "kyc pending what should i do",
        "documents needed for verification",
        "why my kyc rejected",
    ],
    "account_opening": [
        "open savings account online",
        "requirements to open account",
        "can foreigner open bank account",
        "new account registration steps",
    ],
    "bill_payment": [
        "pay electricity bill in app",
        "set monthly water bill auto pay",
        "how to pay internet bill",
        "is there fee for bill payment",
    ],
    "card_issue": [
        "my aba card not working",
        "card declined at store",
        "replace damaged debit card",
        "i lost my card block it",
    ],
    "transaction_status": [
        "payment still pending",
        "where is my transfer",
        "did my transaction go through",
        "status of my latest payment",
    ],
    "fee_inquiry": [
        "what is transfer fee",
        "atm withdrawal charge amount",
        "why was i charged fee",
        "international payment fee",
    ],
    "cash_withdrawal": [
        "how to withdraw cash from atm",
        "atm cash withdrawal limit",
        "cash withdrawal failed",
        "did not receive atm cash",
    ],
    "wallet_topup": [
        "top up wing wallet with bank card",
        "add money to truemoney wallet",
        "wallet top up pending",
        "wallet top up failed",
    ],
}
