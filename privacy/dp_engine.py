# # privacy/dp_engine.py

# from opacus import PrivacyEngine


# class DPEngine:

#     def __init__(
#         self,
#         noise_multiplier=1.0,
#         max_grad_norm=1.0,
#         delta=1e-5,
#     ):

#         self.noise_multiplier = noise_multiplier
#         self.max_grad_norm = max_grad_norm
#         self.delta = delta

#         self.privacy_engine = PrivacyEngine()

#     # -------------------------------------------------
#     # Make Model Private
#     # -------------------------------------------------
#     def make_private(
#         self,
#         model,
#         optimizer,
#         data_loader,
#     ):

#         model, optimizer, data_loader = (
#             self.privacy_engine.make_private(
#                 module=model,
#                 optimizer=optimizer,
#                 data_loader=data_loader,
#                 noise_multiplier=self.noise_multiplier,
#                 max_grad_norm=self.max_grad_norm,
#             )
#         )

#         return model, optimizer, data_loader

#     # -------------------------------------------------
#     # Get Privacy Budget
#     # -------------------------------------------------
#     def get_epsilon(self):

#         epsilon = self.privacy_engine.get_epsilon(
#             delta=self.delta
#         )

#         return epsilon

# =====================================================
# privacy/dp_engine.py
# =====================================================

from opacus import PrivacyEngine


class DPEngine:

    def __init__(

        self,

        noise_multiplier,

        max_grad_norm,

        delta
    ):

        self.noise_multiplier = noise_multiplier

        self.max_grad_norm = max_grad_norm

        self.delta = delta

        self.privacy_engine = PrivacyEngine()

    # =================================================
    # MAKE MODEL PRIVATE
    # =================================================
    def make_private(

        self,

        model,

        optimizer,

        data_loader
    ):

        model, optimizer, data_loader = (

            self.privacy_engine.make_private(

                module=model,

                optimizer=optimizer,

                data_loader=data_loader,

                noise_multiplier=self.noise_multiplier,

                max_grad_norm=self.max_grad_norm
            )
        )

        return (
            model,
            optimizer,
            data_loader
        )

    # =================================================
    # GET EPSILON
    # =================================================
    def get_epsilon(self):

        epsilon = self.privacy_engine.accountant.get_epsilon(
            delta=self.delta
        )

        return epsilon