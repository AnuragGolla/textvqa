# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from pythia.common.registry import registry
from pythia.models.pythia import Pythia
from pythia.modules.layers import ClassifierLayer


@registry.register_model("lorra")
class LoRRA(Pythia):
    def __init__(self, config):
        super().__init__(config)

    def build(self):
        self._init_text_embeddings("text")
        # For LoRRA context feature and text embeddings would be identity
        # but to keep a unified API, we will init them also
        # and we need to build them first before building pythia's other
        # modules as some of the modules require context attributes to be set
        self._init_text_embeddings("context")
        self._init_feature_encoders("context")
        self._init_feature_embeddings("context")
        super().build()

    def get_optimizer_parameters(self, config):
        params = super().get_optimizer_parameters(config)
        params += [
            {"params": self.context_feature_embeddings_list.parameters()},
            {"params": self.context_embeddings.parameters()},
            {"params": self.context_feature_encoders.parameters()},
        ]

        return params

    # def _get_classifier_input_dim(self):
    #     # Now, the classifier's input will be cat of image and context based
    #     # features
    #     return 2 * super()._get_classifier_input_dim()

    def forward(self, sample_list):
        sample_list.text = self.word_embedding(sample_list.text)
        # print("sample_list.text\t{}".format(sample_list.text.shape))
        

        text_embedding_total = self.process_text_embedding(sample_list)
        # print("text_embedding_total\t{}".format(text_embedding_total.shape))

        
        image_embedding_total, _ = self.process_feature_embedding(
            "image", sample_list, text_embedding_total
        )
        # print("image_embedding_total\t{}".format(image_embedding_total.shape))


        context_embedding_total, _ = self.process_feature_embedding(
            "context", sample_list, text_embedding_total, ["order_vectors"]
        )
        # print("context_embedding_total\t{}".format(context_embedding_total.shape))




        joint_lang_embedding = self.combine_embeddings(
            ["text", "context"],
            [text_embedding_total, context_embedding_total],
        )
        # print("joint_lang_embedding\t{}".format(joint_lang_embedding.shape))



        if self.inter_model is not None:
            # print("inter_model\t{}".format(True))
            image_embedding_total = self.inter_model(image_embedding_total)
            # print("image_embedding_total\t{}".format(image_embedding_total.shape))
        else:
            # print("inter_model\t{}".format(False))
            # print("image_embedding_total\t{}".format(image_embedding_total.shape))
            pass

        joint_embedding = self.combine_embeddings(
            ["image", "text"],
            [image_embedding_total, text_embedding_total, context_embedding_total],
        )
        # print("joint_embedding\t{}".format(joint_embedding.shape))



        joint_pairwise_embedding = self.combine_embeddings(
            ["image", "context", "lang", "text"],
            [image_embedding_total, context_embedding_total, joint_lang_embedding, text_embedding_total],
        )
        # print("joint_pairwise_embedding\t{}".format(joint_pairwise_embedding.shape))



        scores = self.calculate_logits(joint_pairwise_embedding)
        # print("scores\t{}".format(scores.shape))

        # print("----------------------------------------------------------")
        return {"scores": scores}




