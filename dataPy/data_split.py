import pandas as pd

X_column=["deal_time","PTMYEAR","TERMNOTE1","TERMIFEXERCISE","ISSUERUPDATED",
          "LATESTPAR","ISSUEAMOUNT","OUTSTANDINGBALANCE","LATESTISSURERCREDITRATING",
          "RATINGOUTLOOKS","RATE_RATEBOND","RATE_RATEBOND2","RATE_LATESTMIR_CNBD",
          "INTERESTTYPE","COUPONRATE","COUPONRATE2","PROVINCE","CITY","MUNICIPALBOND",
          "WINDL2TYPE","SUBORDINATEORNOT","PERPETUALORNOT","PRORNOT","INDUSTRY_SW",
          "ISSUE_ISSUEMETHOD","EXCH_CITY","TAXFREE","MULTIMKTORNOT","IPO_DATE",
          "MATURITYDATE","NXOPTIONDATE","NATURE1","AGENCY_GRNTTYPE","AGENCY_GUARANTOR",
          "RATE_RATEGUARANTOR","LISTINGORNOT1","EMBEDDEDOPT","CLAUSEABBR"
          ]
y_column=["yield"]

def xy_split(bond_pd,X_column,y_column):
    sets=["train","valid","test"]
    X_ls=[]
    y_ls=[]
    for id,item in bond_pd.iterrows():
        X=item[X_column]
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