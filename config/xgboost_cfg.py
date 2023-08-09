from yacs.config import CfgNode as CN

_C = CN()
_C.PARAM = CN()
_C.PARAM.MAX_DEPTH = 12
_C.PARAM.ETA = 0.1
_C.PARAM.OBJECTIVE = "reg:squarederror"
_C.PARAM.EVAL_METRIC = "mae"
_C.PARAM.TREE_METHOD = "hist"
_C.PARAM.DEVICE = "cuda"
_C.PARAM.SUBSAMPLE = 1
_C.PARAM.ALPHA = 0.5
_C.PARAM.VERBOSITY = 0
_C.PARAM.COLSAMPLE_BYTREE = 0.7
_C.EARLY_STOPPING  =  1
_C.NUM_ROUND  =  3500
_C.EVAL_NAME = "valid"
_C.EARLY_STOP_ROUNDS = 15
_C.TRAIN_PATH = "dealed_dir\\sets_split0808region\\train.csv"
_C.VALID_PATH = "dealed_dir\\sets_split0808region\\valid.csv"
_C.TEST_PATH =  "dealed_dir\\sets_split0808region\\test.csv"
_C.MODEL_SAVE = "res"
_C.MODEL_TAG = "model\\xgboost"
# ,"date"
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
# "time_diff-1","time_diff-2","time_diff-3","time_diff-4","time_diff-5",
#"yield-1","yield-2","yield-3","yield-4","yield-5",
# "termnote2","termnote3"
# ,"yield-4","yield-5",
#           "GDP","GENERAL_BUDGET_MONEY","SSSR_RADIO","CZZJL",
#             "ZFXJJ_MONEY","ZFZWYE","QYFZCTYX","QYFZCTCXZ","QYFZCTCXZ_RADIO",
#             "CTYXZWZS","CTYXZWBS"
_C.Y_COLUMN=["yield"]

cfg = _C