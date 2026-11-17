import azure.functions as func
from difflib import SequenceMatcher
import json

app = func.FunctionApp()


fake_employees = [
    {"name": "Alice Sinclair", "title": "Senior Loan Officer"},
    {"name": "Bob Johnson", "title": "Junior Loan Officer"},
    {"name": "Carol White", "title": "Loan Processor"},
    {"name": "David Miller", "title": "Mortgage Consultant"},
    {"name": "Emma Davis", "title": "Loan Officer"},
    {"name": "Frank Wilson", "title": "Senior Loan Processor"},
    {"name": "Grace Martinez", "title": "Mortgage Advisor"},
    {"name": "Henry Anderson", "title": "Loan Underwriter"},
    {"name": "Isabella Thomas", "title": "Loan Officer"},
    {"name": "Jack Taylor", "title": "Mortgage Specialist"},
    {"name": "Katherine Moore", "title": "Loan Processor"},
    {"name": "Liam Jackson", "title": "Senior Loan Officer"},
    {"name": "Mia Martin", "title": "Mortgage Consultant"},
    {"name": "Noah Lee", "title": "Loan Underwriter"},
    {"name": "Olivia Perez", "title": "Loan Officer"},
    {"name": "Peter Thompson", "title": "Junior Loan Officer"},
    {"name": "Quinn Garcia", "title": "Mortgage Advisor"},
    {"name": "Ruby Harris", "title": "Loan Processor"},
    {"name": "Samuel Clark", "title": "Mortgage Specialist"},
    {"name": "Tessa Lewis", "title": "Senior Loan Processor"},
    {"name": "Umar Robinson", "title": "Loan Officer"},
    {"name": "Violet Walker", "title": "Mortgage Consultant"},
    {"name": "William Young", "title": "Loan Underwriter"},
    {"name": "Xavier Allen", "title": "Junior Loan Officer"},
    {"name": "Yara King", "title": "Mortgage Advisor"},
    {"name": "Zachary Scott", "title": "Loan Processor"},
    {"name": "Avery Green", "title": "Senior Loan Officer"},
    {"name": "Sandeep Agarwal", "title": "Senior Loan Officer"},
    {"name": "Blake Baker", "title": "Mortgage Specialist"},
]


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="search_employees",
    description="Find the three closest employees by name, title, or both. At least one is required.",
    tool_properties=json.dumps([
        {
            "propertyName": "name",
            "propertyType": "string",
            "description": "Employee name to search for.",
            "isRequired": False,
        },
        {
            "propertyName": "title",
            "propertyType": "string",
            "description": "Employee title to search for.",
            "isRequired": False,
        },
    ]),
)
def search_employees(context: str) -> str:
    arguments = json.loads(context)["arguments"]
    search_terms = {}
    for field in ("name", "title"):
        if field in arguments:
            value = arguments[field]
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string")
            value = value.strip().casefold()
            if value:
                search_terms[field] = value

    if not search_terms:
        raise ValueError("Provide at least one non-empty name or title")

    def score(employee):
        return sum(
            SequenceMatcher(None, value, employee[field].casefold()).ratio()
            for field, value in search_terms.items()
        ) / len(search_terms)

    matches = sorted(fake_employees, key=score, reverse=True)[:3]
    return json.dumps(matches)


fake_customers = [
    {
        "name": "Joe Obarzanek", 
        "email": "joe.obarzanek@example.com", 
        "phone": "734-555-1234", 
        "loan-status": "in-progress", 
        "loan-details": "Customer needs to sign and submit their documents. If they haven't been received, follow up with the customer."
    },
    {
        "name": "Maya Chen",
        "email": "maya.chen@example.com",
        "phone": "202-555-0101",
        "loan-status": "in-progress",
        "loan-details": "Customer needs to upload proof of income. Follow up if the documents have not been received."
    },
    {
        "name": "Ethan Brooks",
        "email": "ethan.brooks@example.com",
        "phone": "303-555-0102",
        "loan-status": "pending-review",
        "loan-details": "Customer's application is awaiting review. Notify the customer when the review is complete."
    },
    {
        "name": "Sofia Patel",
        "email": "sofia.patel@example.com",
        "phone": "415-555-0103",
        "loan-status": "in-progress",
        "loan-details": "Customer needs to provide a recent bank statement. Follow up if it has not been submitted."
    },
    {
        "name": "Lucas Rivera",
        "email": "lucas.rivera@example.com",
        "phone": "512-555-0104",
        "loan-status": "approved",
        "loan-details": "Loan has been approved. Customer should review and sign the closing documents."
    },
    {
        "name": "Ava Thompson",
        "email": "ava.thompson@example.com",
        "phone": "617-555-0105",
        "loan-status": "in-progress",
        "loan-details": "Customer needs to confirm their employment details. Contact them if confirmation is still outstanding."
    },
    {
        "name": "Noah Williams",
        "email": "noah.williams@example.com",
        "phone": "702-555-0106",
        "loan-status": "pending-review",
        "loan-details": "The submitted documents are under review. Follow up with the customer if additional information is needed."
    },
    {
        "name": "Isabella Garcia",
        "email": "isabella.garcia@example.com",
        "phone": "801-555-0107",
        "loan-status": "in-progress",
        "loan-details": "Customer needs to submit a signed disclosure form. Follow up if it has not been received."
    },
    {
        "name": "Oliver Martin",
        "email": "oliver.martin@example.com",
        "phone": "919-555-0108",
        "loan-status": "closed",
        "loan-details": "Loan file is complete and closed. No further action is currently required."
    },
    {
        "name": "Amara Okafor",
        "email": "amara.okafor@example.com",
        "phone": "206-555-0109",
        "loan-status": "in-progress",
        "loan-details": "Customer needs to confirm the requested loan amount. Follow up if confirmation has not been received."
    },
    {
        "name": "Caleb Foster",
        "email": "caleb.foster@example.com",
        "phone": "404-555-0110",
        "loan-status": "pending-review",
        "loan-details": "Customer's application is waiting for underwriting review. Provide an update when the review is complete."
    },
]


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="verify_identity",
    description="Verify the identity of the customer caller.",
    tool_properties=json.dumps([
        {
            "propertyName": "account_code",
            "propertyType": "string",
            "description": "Verification code to check the customer's identity.",
            "isRequired": False,
        },
    ]),
)
def verify_identity(context: str):
    arguments = json.loads(context)["arguments"]
    code = arguments.get("account_code")
    if code:
        return fake_customers[0]
    return {}


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="update_loan_status",
    description="Update the loan status of a customer.",
    tool_properties=json.dumps([
        {
            "propertyName": "email",
            "propertyType": "string",
            "description": "Email of the customer.",
            "isRequired": True,
        },
        {
            "propertyName": "loan_status",
            "propertyType": "string",
            "description": "New loan status to set for the customer.",
            "isRequired": True,
        },
    ]),
)
def update_loan_status(context: str):
    arguments = json.loads(context)["arguments"]
    email = arguments.get("email")
    new_status = arguments.get("loan_status")
    for customer in fake_customers:
        if customer["email"] == email:
            customer["loan-status"] = new_status
            return {"message": f"successfully set the customer's loan status to {new_status}"}
    return {"message": "customer not found"}


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="send_loan_details",
    description="Send the loan details of a customer.",
    tool_properties=json.dumps([
        {
            "propertyName": "email",
            "propertyType": "string",
            "description": "Email of the customer.",
            "isRequired": True,
        },
    ]),
)
def send_loan_details(context: str):
    arguments = json.loads(context)["arguments"]
    email = arguments.get("email")
    for customer in fake_customers:
        if customer["email"] == email:
            return {"message": f"Sent loan details to the customer with email {email}"}
    return {"message": "Customer with email {} not found".format(email)}


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="get_loan_details",
    description="Get the loan details of a customer.",
    tool_properties=json.dumps([
        {
            "propertyName": "email",
            "propertyType": "string",
            "description": "Email of the customer.",
            "isRequired": True,
        },
    ]),
)
def get_loan_details(context: str):
    arguments = json.loads(context)["arguments"]
    email = arguments.get("email")
    for customer in fake_customers:
        if customer["email"] == email:
            return {"message": "We have received the necessary documents and signatures from the customer!"}
    return {"message": "Customer with email {} not found".format(email)}


@app.mcp_tool_trigger(
    arg_name="context",
    tool_name="transfer_to_loan_officer",
    description="Transfer a customer to the loan officer.",
    tool_properties=json.dumps([
        {
            "propertyName": "email",
            "propertyType": "string",
            "description": "Email of the customer.",
            "isRequired": True,
        },
        {
            "propertyName": "loan_officer_name",
            "propertyType": "string",
            "description": "Name of the loan officer.",
            "isRequired": True,
        },
    ]),
)
def transfer_to_loan_officer(context: str):
    arguments = json.loads(context)["arguments"]
    email = arguments.get("email")
    for customer in fake_customers:
        if customer["email"] == email:
            loan_officer_name = arguments.get("loan_officer_name")
            return {"message": f"Transferred the customer with email {email} to the loan officer {loan_officer_name}."}
    return {"message": "Customer with email {} not found".format(email)}