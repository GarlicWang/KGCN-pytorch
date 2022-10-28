def get_model_data_name(args):
    if not args.relabel:
        if args.only_stock_rel:
            if args.stock_article_only:
                run_id = "d3a383a83143404da6d48283f829d9dc" # global_neg V3_stock_rel_stock_article
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_stock_rel_stock_article'
                print("V3_stock_rel_stock_article")
            else:
                run_id = "a25d8e8d943442fb91177fb75eb2862e" # global_neg V3_stock_rel
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_stock_rel'
                print("V3_stock_rel")
        else:
            if args.stock_article_only:
                run_id = "9f6e83eb51a34923a6883b6de1963e29" # global_neg V3_two_rel_stock_article
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_two_rel_stock_article'
                print("V3_two_rel_stock_article")
            else:
                run_id = "afe396c21df8417eb52985bc359c0925" # global_neg V3_two_rel
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_two_rel'
                print("V3_two_rel")
    else:
        if args.only_stock_rel:
            if args.stock_article_only:
                run_id = "23682dce93704d1ab66c17ac0f0d15bd" # hard_neg V3_stock_rel_stock_article
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_relabel_stock_rel_stock_article'
                print("relabel V3_stock_rel_stock_article")
            else:
                run_id = "63000f1a54c64263aa42a11b91b27496"
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_relabel_stock_rel'
                print("relabel V3_stock_rel")
        else:
            if args.stock_article_only:
                run_id = "16e5fa1d6c7a4ffd8fa6d7cf0b6ff62d" 
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_relabel_two_rel_stock_article'
                print("relabel V3_two_rel_stock_article")
            else:
                run_id = "c74dfd30ad5e4b008159a48812192224"
                model_uri = f'runs:/{run_id}/KGCN_epoch_3'
                data_version = 'CMoney_relabel_two_rel'
                print("relabel V3_two_rel")
    return model_uri, data_version