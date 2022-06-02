from dataloader import Dataset

if __name__ == '__main__':
    dataset = Dataset(conv_init_file="./data/guanpian/conv_init.csv",
                      parallel_info_file="./data/guanpian/parallel_info.json")

