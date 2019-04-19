from model.data_utils import CoNLLDataset, get_processing_word
from model.config import Config

from model.ripple_model import RippleModel


def main():
    # create instance of config，这里的config实现了load data的作用
    #拥有词表、glove训练好的embeddings矩阵、str->id的function
    config = Config()


    # build model
    model = RippleModel(config)
    model.build("train")


    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets [(char_ids), word_id]
    processing_word = get_processing_word(lowercase=True)
    dev = CoNLLDataset(config.filename_dev, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)
    test = CoNLLDataset(config.filename_test, processing_word)

    train4rp = CoNLLdata4ripple(train, processing_word=config.processing_word,
                                    processing_tag=config.processing_tag)
    dev4rp = CoNLLdata4ripple(dev, processing_word=config.processing_word,
                                  processing_tag=config.processing_tag)
    test4rp = CoNLLdata4ripple(test, processing_word=config.processing_word,
                                   processing_tag=config.processing_tag)

    # train model
    model.train(train, dev)




if __name__ == "__main__":
    main()
