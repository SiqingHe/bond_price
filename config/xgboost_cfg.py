from yacs.config import CfgNode as CN

_C = CN()
_C.PARAM = CN()
_C.PARAM.MAX_DEPTH = 10
_C.PARAM.ETA = 0.12
_C.PARAM.OBJECTIVE = "reg:squarederror"
_C.PARAM.EVAL_METRIC = "mae"
_C.PARAM.TREE_METHOD = "hist"
_C.PARAM.DEVICE = "cuda"
_C.PARAM.SUBSAMPLE = 1
_C.PARAM.ALPHA = 1
_C.PARAM.VERBOSITY = 0
_C.PARAM.COLSAMPLE_BYTREE = 0.8
_C.EARLY_STOPPING  =  1
_C.NUM_ROUND  =  3500
_C.EVAL_NAME = "valid"
_C.EARLY_STOP_ROUNDS = 15
_C.TRAIN_PATH = "dealed_dir\\sets_split0804\\train.csv"
_C.VALID_PATH = "dealed_dir\\sets_split0804\\valid.csv"
_C.TEST_PATH =  "dealed_dir\\sets_split0804\\test.csv"
_C.MODEL_SAVE = "model\\xgboost\\res"
_C.X_COLUMN = ["deal_time","date","PTMYEAR","TERMNOTE1","TERMIFEXERCISE","ISSUERUPDATED",
          "LATESTPAR","ISSUEAMOUNT","OUTSTANDINGBALANCE","LATESTISSURERCREDITRATING",
          "RATINGOUTLOOKS","RATE_RATEBOND","RATE_RATEBOND2","RATE_LATESTMIR_CNBD",
          "INTERESTTYPE","COUPONRATE","COUPONRATE2","PROVINCE","CITY","MUNICIPALBOND",
          "WINDL2TYPE","SUBORDINATEORNOT","PERPETUALORNOT","PRORNOT","INDUSTRY_SW",
          "ISSUE_ISSUEMETHOD","EXCH_CITY","TAXFREE","MULTIMKTORNOT","IPO_DATE",
          "MATURITYDATE","NXOPTIONDATE","NATURE1","AGENCY_GRNTTYPE","AGENCY_GUARANTOR",
          "RATE_RATEGUARANTOR","LISTINGORNOT1","EMBEDDEDOPT","CLAUSEABBR_0",
          "CLAUSEABBR_1","CLAUSEABBR_2","CLAUSEABBR_3","CLAUSEABBR_4",
          "CLAUSEABBR_5","CLAUSEABBR_6","termnote2","termnote3","time_diff",
          "time_diff-1","time_diff-2","time_diff-3","time_diff-4","time_diff-5",
          "yd*diff-1",
          "yield-1","yield-2","yield-3","yield-4","yield-5",
          "GDP","GENERAL_BUDGET_MONEY","SSSR_RADIO","CZZJL",
            "ZFXJJ_MONEY","ZFZWYE","QYFZCTYX","QYFZCTCXZ","QYFZCTCXZ_RADIO",
            "CTYXZWZS","CTYXZWBS"
          ]
_C.DTYPE =[
           ("deal_time",float),
           ("date",float),
           ("PTMYEAR",float),
           ("TERMNOTE1",float),
           ("TERMIFEXERCISE",float),
           ("ISSUERUPDATED",int),
         ("LATESTPAR",float),
         ("ISSUEAMOUNT",float),
         ("OUTSTANDINGBALANCE",float),
         ("LATESTISSURERCREDITRATING",int),
         ("RATINGOUTLOOKS",int),
         ("RATE_RATEBOND",int),
         ("RATE_RATEBOND2",int),
         ("RATE_LATESTMIR_CNBD",int),
         ("INTERESTTYPE",int),
         ("COUPONRATE",float),
         ("COUPONRATE2",float),
         ("PROVINCE",int),
         ("CITY",int),
         ("MUNICIPALBOND",int),
         ("WINDL2TYPE",int),
         ("SUBORDINATEORNOT",int),
         ("PERPETUALORNOT",int),
         ("PRORNOT",int),
         ("INDUSTRY_SW",int),
         ("ISSUE_ISSUEMETHOD",int),
         ("EXCH_CITY",int),
         ("TAXFREE",int),
         ("MULTIMKTORNOT",int),
         ("IPO_DATE",int),
         ("MATURITYDATE",int),
         ("NXOPTIONDATE",int),
         ("NATURE1",int),
         ("AGENCY_GRNTTYPE",int),
         ("AGENCY_GUARANTOR",int),
         ("RATE_RATEGUARANTOR",int),
         ("LISTINGORNOT1",int),
         ("EMBEDDEDOPT",int),
         ("CLAUSEABBR_0",int),
         ("CLAUSEABBR_1",int),
         ("CLAUSEABBR_2",int),
         ("CLAUSEABBR_3",int),
         ("CLAUSEABBR_4",int),
         ("CLAUSEABBR_5",int),
         ("CLAUSEABBR_6",int),
         ("termnote2",float),
         ("termnote3",int),
         ("time_diff",float),
         ("time_diff-1",float),
         ("time_diff-2",float),
         ("time_diff-3",float),
         ("time_diff-4",float),
         ("time_diff-5",float),
         ("yd*diff-1",float),
         ("yield-1",float),
         ("yield-2",float),
         ("yield-3",float),
         ("yield-4",float),
         ("yield-5",float),
         ("GDP",float),
         ("GENERAL_BUDGET_MONEY",float),
         ("SSSR_RADIO",float),
         ("CZZJL",float),
           ("ZFXJJ_MONEY",float),
           ("ZFZWYE",float),
           ("QYFZCTYX",float),
           ("QYFZCTCXZ",float),
           ("QYFZCTCXZ_RADIO",float),
           ("CTYXZWZS",float),
           ("CTYXZWBS",float)
]
#"yield-1","yield-2","yield-3","yield-4","yield-5",
# "termnote2","termnote3"
_C.Y_COLUMN=["yield"]

cfg = _C