import pandas as pd

def main():
    csv_path = 'output/submission.csv'
    xlsx_path = 'output/submission.xlsx'
    
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Writing {xlsx_path}...")
    df.to_excel(xlsx_path, index=False, engine='openpyxl')
    print("Conversion successful.")

if __name__ == "__main__":
    main()
