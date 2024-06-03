HEURISTIC_WEIGHT = {
    "llama-7b":{
            "wq": {"c_all":0.602834665,
                    "c_all_sum":0.549783276,
                    "gc":0.603317881,
                    "gc_sum":0.557985577,
                    "gc_w":0.60935737,
                    "gc_w_sum":0.561693931,
                    "gendoc":0.550378486,
                    "gg":0.564828793,
                    "gg_sum":0.5277845,
                    "gm":0.592946156,
                    "gm_sum":0.54799379,
                    "gm_w":0.610902674,
                    "gm_w_sum":0.541743047,
                    "query_only":0.565378789,
                    "wiki":0.616722245,
                    "wiki_sum":0.554145865},

            "nq":  {"c_all": 0.576433,
                    "c_all_sum": 0.552392,
                    "gc": 0.624559,
                    "gc_sum": 0.589006,
                    "gc_w": 0.628043,
                    "gc_w_sum": 0.594947,
                    "gendoc": 0.4316  ,
                    "gg": 0.570783,
                    "gg_sum": 0.534789,
                    "gm": 0.611518,
                    "gm_sum": 0.567084,
                    "gm_w" : 0.62286 ,
                    "gm_w_sum": 0.562967,
                    "query_only": 0.422837,
                    "wiki": 0.569146,
                    "wiki_sum": 0.524521},

            "tq": {
                    "c_all": 0.749130433,
                    "c_all_sum": 0.729848766,
                    "gc": 0.795997772,
                    "gc_sum": 0.77102996,
                    "gc_w": 0.80866645,
                    "gc_w_sum": 0.76852698,
                    "gendoc": 0.612197532,
                    "gg": 0.764767231,
                    "gg_sum": 0.737413083,
                    "gm": 0.791511335,
                    "gm_sum": 0.755196323,
                    "gm_w": 0.798845202,
                    "gm_w_sum": 0.742413721,
                    "query_only": 0.623642895,
                    "wiki": 0.701507151,
                    "wiki_sum": 0.661401904}
        },
    "llama-13b": {
        "wq": {"c_all":0.656051975,
               "c_all_sum":0.542856062,
               "gc":0.665165568,
               "gc_sum":0.544663824,
               "gc_w":0.679439397,
               "gc_w_sum":0.558953215,
               "gendoc":0.582777186,
               "gg":0.633411578,
               "gg_sum":0.54098936
,              "gm":0.660296417,
               "gm_sum":0.548385644,
               "gm_w":0.680228726,
               "gm_w_sum":0.561529412,
               "query_only":0.642986514,
               "wiki":0.685672412,
               "wiki_sum":0.559140208},
        "nq": {
            "c_all": 0.667833891,
            "c_all_sum": 0.573906921,
            "gc": 0.685814558,
            "gc_sum": 	0.60492379,
            "gc_w": 0.701922469,
            "gc_w_sum": 0.616539881,
            "gendoc": 0.503661149,
            "gg": 0.624988908,
            "gg_sum": 0.561530982,
            "gm": 0.675922707,
            "gm_sum": 0.598243338,
            "gm_w": 0.701466512,
            "gm_w_sum": 0.621179494,
            "query_only": 0.558839546,
            "wiki": 0.646527776,
            "wiki_sum": 0.556288599,
        },
        "tq": {
            "c_all": 0.835449049,
            "c_all_sum": 0.768437926,
            "gc": 0.854508128,
            "gc_sum": 0.793252315,
            "gc_w": 0.864271304,
            "gc_w_sum": 0.800820009,
            "gendoc": 0.713468412,
            "gg": 0.821135067,
            "gg_sum": 0.765579398,
            "gm": 0.847256581,
            "gm_sum": 0.783478083,
            "gm_w": 0.856076107,
            "gm_w_sum": 0.79358161,
            "query_only": 0.736710376,
            "wiki": 0.792213552,
            "wiki_sum": 0.70657155,
        }
    } ,
    "turbo": {
        "wq": {
            "c_all":0.634144839,
            "c_all_sum":0.611443512,
            "gc":	0.62133721,
            "gc_sum":0.587215344,
            "gc_w":0.619269025,
            "gc_w_sum":0.579488219,
            "gendoc":0.613198511,
            "gg":0.592573615,
            "gg_sum":0.546744135,
            "gm":0.610354526,
            "gm_sum": 0.55663166,
            "gm_w":0.627570485,
            "gm_w_sum":0.573478243,
            "query_only":0.607017643,
            "system_answer":0.632586456,
            "wiki":0.643892129,
            "wiki_sum":0.598915196,
        },
        "nq":{
            "c_all":	0.645920218,
            "c_all_sum":	0.644973563,
            "gc":	0.667197785,
            "gc_sum":	0.632845155,
            "gc_w":	0.688377588,
            "gc_w_sum":	0.628497116,
            "gendoc":	0.628737844,
            "gg":	0.622772349,
            "gg_sum":	0.603118519,
            "gm":	0.655366988,
            "gm_sum":	0.604209286,
            "gm_w":	0.685028157,
            "gm_w_sum":	0.627001505,
            "query_only":	0.622459336,
            "wiki":	0.681898343,
            "wiki_sum":	0.629018626,
        },
        "tq":{
            "c_all":0.845313813,
            "c_all_sum":0.830763706,
            "gc":	0.85559508,
            "gc_sum":0.829040204,
            "gc_w":0.854854525,
            "gc_w_sum":0.827390726,
            "gendoc":0.813072703,
            "gg":0.823693578,
            "gg_sum":0.791952513,
            "gm":0.851484479,
            "gm_sum":0.822945926,
            "gm_w":0.847892385,
            "gm_w_sum":	0.82236537,
            "query_only":0.827337415,
            "wiki":0.805024395,
            "wiki_sum":0.743632111
        }
    }  
}