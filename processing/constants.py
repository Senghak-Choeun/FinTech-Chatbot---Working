INTENT_RESPONSE_TEMPLATE = {
    "balance_inquiry": "You can check your latest balance in the app dashboard under Accounts. If the balance looks delayed, refresh the page or pull down to sync.",
    "fund_transfer": "To transfer funds, select recipient, amount, and confirm with secure verification. Keep your OTP private before completing the transfer.",
    "fraud_report": "Please report the suspicious transaction immediately and lock your card for safety. Our team can then start a dispute review.",
    "loan_info": "Loan options and eligibility can be reviewed in the Loans section. You can compare tenure, rates, and repayment amount before applying.",
    "kyc_verification": "KYC verification requires identity details and document confirmation. Ensure your uploaded ID is clear and not expired.",
    "account_opening": "You can open an account online by completing the onboarding steps. Prepare your ID and contact details to avoid delays.",
    "bill_payment": "Bill payments are available in Payments and Bills, including recurring setup. You can set reminders so due dates are not missed.",
    "card_issue": "For card issues, please open Card Support to freeze, replace, or troubleshoot. Use instant freeze first if you suspect risk.",
    "transaction_status": "You can track the latest transaction status in your transaction history. Pending transfers usually update after bank settlement windows.",
    "fee_inquiry": "Fees depend on account plan and transaction type. Check fee preview before confirmation to avoid surprise charges.",
    "cash_withdrawal": "You can withdraw cash using card-enabled ATM services supported by your account. Daily limits and partner ATM fees may apply.",
    "wallet_topup": "Wallet top up is available through linked bank card or bank transfer options. If top up is pending, check card limits and network status.",
}

INTENT_RESPONSE_VARIANTS = {
    "balance_inquiry": [
        "Open Accounts in the app to view your current and available balance. Refresh once if your latest credit is not visible yet.",
        "Your balance is shown on the dashboard under Accounts. If the amount looks old, wait a moment and sync again.",
        "Go to the account summary page to check your updated balance. Recent transactions can take a short time to post.",
    ],
    "fund_transfer": [
        "Choose Transfer, select the recipient, and enter the amount. Confirm with OTP or biometric verification to finish securely.",
        "Start a new transfer from the Payments section and verify beneficiary details carefully. Approve the transaction using your security step.",
        "Use the transfer flow in-app to send funds to bank or wallet accounts. Keep your verification code private before confirming.",
    ],
    "fraud_report": [
        "Please freeze your card or account access immediately from security settings. Then submit a fraud report so we can investigate the transaction.",
        "If you do not recognize this payment, lock your card first to stop further risk. After that, file a dispute report in the app.",
        "Open security support and mark the transaction as unauthorized. We will review the case and guide you through next steps.",
    ],
    "loan_info": [
        "Loan choices are available in the Loans section with rates and repayment terms. Review eligibility before submitting your request.",
        "You can compare loan amount, tenor, and monthly installment directly in-app. Apply after confirming the terms that fit your budget.",
        "Visit Loans to see available products and approval requirements. Keep your profile and income details updated for faster checks.",
    ],
    "kyc_verification": [
        "Complete KYC by submitting your ID details and clear document photos. Make sure the name and date of birth match exactly.",
        "KYC review needs valid identity documents and profile confirmation. Blurry images or expired IDs can delay approval.",
        "Go to verification settings and upload the required documents. You can track KYC progress in the same section.",
    ],
    "account_opening": [
        "You can open a new account online through the onboarding flow. Keep your ID and phone number ready before starting.",
        "Start account opening in the app and fill in personal details accurately. Verification steps must be completed to activate the account.",
        "New account registration is available digitally with quick KYC checks. Follow each step and review your information before submitting.",
    ],
    "bill_payment": [
        "Pay bills from the Payments and Bills section in a few taps. You can also enable recurring payments for monthly utilities.",
        "Select the biller, enter the reference number, and confirm payment securely. Turn on reminders to avoid late fees.",
        "Bill payment supports common services like electricity, water, and internet. Check the fee preview before final confirmation.",
    ],
    "card_issue": [
        "Open Card Support to freeze, unblock, replace, or troubleshoot card problems. Use instant freeze first if the card is lost.",
        "For declined or damaged cards, go to card management in the app. You can request a replacement and track delivery status.",
        "Card issues are handled from the support section with guided actions. Keep your alerts enabled for card security updates.",
    ],
    "transaction_status": [
        "Check transaction history to see whether the payment is pending, completed, or failed. Some transfers update after clearing windows.",
        "You can view real-time transaction progress in the app activity tab. If it remains pending too long, contact support with the reference ID.",
        "Open your latest transfer details to track processing status. Cross-bank transfers may need extra settlement time.",
    ],
    "fee_inquiry": [
        "Fees depend on your plan and transaction type. Review the fee breakdown shown before you confirm payment.",
        "Open pricing or transfer preview to see service charges in advance. Card, ATM, or cross-border actions may have different fees.",
        "The app displays applicable charges during checkout and transfers. Please verify fee details before final submission.",
    ],
    "cash_withdrawal": [
        "You can withdraw cash at supported ATMs using your linked card. Daily limits and operator fees may apply.",
        "Cash withdrawal is available through partner ATM networks. If a withdrawal fails, check limits and card status first.",
        "Use card-enabled ATMs to withdraw funds from your account. Keep the receipt and reference if the cash was not dispensed.",
    ],
    "wallet_topup": [
        "Top up your wallet using a linked bank card or transfer option. Pending top ups usually resolve after payment network confirmation.",
        "Go to Wallet Top Up, choose funding source, and confirm securely. Check card limits if the top up does not complete.",
        "Wallet funding supports common local payment methods in the app. Keep notifications on to confirm top up success quickly.",
    ],
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
        "show my balance lah",
        "check duit left in account",
        "som check balance ban te",
    ],
    "fund_transfer": [
        "send 20 dollars to aba account",
        "transfer money from wing to bank",
        "how to transfer to friend account",
        "bank transfer to local account",
        "transfer now can or not",
        "send duit to maybank account",
        "ot ban transfer to bakong",
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
        "card kena block why",
        "my card cannot tap paynow qr",
        "knyom card ot tvka",
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
        "why got extra charge ah",
        "kena fee for qr payment",
        "fee nis mean ey",
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
        "topup ewallet now can or not",
        "reload wallet cepat please",
        "khqr topup pending",
    ],
}