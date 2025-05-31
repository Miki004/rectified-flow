            with accelerator.accumulate(models_to_accumulate):

                x_1, _ = batch
                x_0 = rectified_flow.sample_source_distribution(x_1.shape[0])
                t = rectified_flow.sample_train_time(x_1.shape[0])
                print(x_0.shape)
                print(x_1.shape) 