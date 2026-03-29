import pandas as pd

def run_pipeline(input_path, output_path):
    df = pd.read_csv(input_path)
    
    df = df.rename(columns={
        "cust_id": "customer_id"
    })
    
    df["transaction_amount"] = df["transaction_amount"].astype(float)
    df = df[df["transaction_amount"] > 0]
    
    df.to_csv(output_path, index=False)
    
    return df