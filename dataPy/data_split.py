import pandas as pd

X_column=["deal_time","PTMYEAR","TERMNOTE1","TERMIFEXERCISE","ISSUERUPDATED",
          "LATESTPAR","ISSUEAMOUNT","OUTSTANDINGBALANCE","LATESTISSURERCREDITRATING",
          "RATINGOUTLOOKS","RATE_RATEBOND","RATE_RATEBOND2","RATE_LATESTMIR_CNBD",
          "INTERESTTYPE","COUPONRATE","COUPONRATE2","PROVINCE","CITY","MUNICIPALBOND",
          "WINDL2TYPE","SUBORDINATEORNOT","PERPETUALORNOT","PRORNOT","INDUSTRY_SW",
          "ISSUE_ISSUEMETHOD","EXCH_CITY","TAXFREE","MULTIMKTORNOT","IPO_DATE",
          "MATURITYDATE","NXOPTIONDATE","NATURE1","AGENCY_GRNTTYPE","AGENCY_GUARANTOR",
          "RATE_RATEGUARANTOR","LISTINGORNOT1","EMBEDDEDOPT","CLAUSEABBR","termnote2",
          "termnote3","yield-1","time_diff"
          ]
y_column=["yield"]

def xy_split(bond_pd,X_column,y_column):
    sets=["train","valid","test"]
    pd_len=bond_pd.shape[0]
    
    X_pd=bond_pd[X_column]
    y_pd=bond_pd[y_column]
    X_ls=[]
    y_ls=[]
    for id,item in bond_pd.iterrows():
        X=item[X_column].to_list()
        y=item[y_column]
        X_ls.append(X)
        y_ls.append(y)
    return X_ls,y_ls

def set2_split():
    pass

def set3_split():
    pass


if __name__=="__main__":
    pass
    test_table=r"D:\python_code\LSTM-master\bond_price\real_data\test\010221.IB_41_test1.csv"
    test_pd=pd.read_csv(test_table,index_col="org_date")
    xy_split(test_pd,X_column,y_column)