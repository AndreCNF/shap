from .deep_pytorch import PyTorchDeepExplainer
from .deep_tf import TFDeepExplainer
from shap.explainers.explainer import Explainer


class DeepExplainer(Explainer):
    """ Meant to approximate SHAP values for deep learning models.

    This is an enhanced version of the DeepLIFT algorithm (Deep SHAP) where, similar to Kernel SHAP, we
    approximate the conditional expectations of SHAP values using a selection of background samples.
    Lundberg and Lee, NIPS 2017 showed that the per node attribution rules in DeepLIFT (Shrikumar,
    Greenside, and Kundaje, arXiv 2017) can be chosen to approximate Shapley values. By integrating
    over many backgound samples DeepExplainer estimates approximate SHAP values such that they sum
    up to the difference between the expected model output on the passed background samples and the
    current model output (f(x) - E[f(x)]).
    """

    def __init__(self, model, data, session=None, learning_phase_flags=None, feedforward_args=None):
        """ An explainer object for a differentiable model using a given background dataset.

        Note that the complexity of the method scales linearly with the number of background data
        samples. Passing the entire training dataset as `data` will give very accurate expected
        values, but be unreasonably expensive. The variance of the expectation estimates scale by
        roughly 1/sqrt(N) for N background data samples. So 100 samples will give a good estimate,
        and 1000 samples a very good estimate of the expected values.

        Parameters
        ----------
        model : if framework == 'tensorflow', (input : [tf.Operation], output : tf.Operation)
             A pair of TensorFlow operations (or a list and an op) that specifies the input and
            output of the model to be explained. Note that SHAP values are specific to a single
            output value, so the output tf.Operation should be a single dimensional output (,1).

            if framework == 'pytorch', an nn.Module object (model), or a tuple (model, layer),
                where both are nn.Module objects
            The model is an nn.Module object which takes as input a tensor (or list of tensors) of
            shape data, and returns a single dimensional output.
            If the input is a tuple, the returned shap values will be for the input of the
            layer argument. layer must be a layer in the model, i.e. model.conv2

        data :
            if framework == 'tensorflow': [numpy.array] or [pandas.DataFrame]
            if framework == 'pytorch': [torch.tensor]
            The background dataset to use for integrating out features. DeepExplainer integrates
            over these samples. The data passed here must match the input operations given in the
            first argument. Note that since these samples are integrated over for each sample you
            should only something like 100 or 1000 random background samples, not the whole training
            dataset.

        if framework == 'tensorflow':

        session : None or tensorflow.Session
            The TensorFlow session that has the model we are explaining. If None is passed then
            we do our best to find the right session, first looking for a keras session, then
            falling back to the default TensorFlow session.

        learning_phase_flags : None or list of tensors
            If you have your own custom learning phase flags pass them here. When explaining a prediction
            we need to ensure we are not in training mode, since this changes the behavior of ops like
            batch norm or dropout. If None is passed then we look for tensors in the graph that look like
            learning phase flags (this works for Keras models). Note that we assume all the flags should
            have a value of False during predictions (and hence explanations).

        if framework == 'pytorch':

        feedforward_args : None or list
            In case your model's feedforward method has additional parameters besides the input data,
            you can specify them in a list format in feedforward_args. This can be useful for instance
            if you're dealing with recurrent neural networks that receive inputs with variable sequence
            length, requiring padding and the list of original sequence lengths. Currently only works
            in PyTorch.
        """
        # first, we need to find the framework
        if type(model) is tuple:
            a, b = model
            try:
                a.named_parameters()
                self.framework = 'pytorch'
            except:
                self.framework = 'tensorflow'
        else:
            try:
                model.named_parameters()
                self.framework = 'pytorch'
            except:
                self.framework = 'tensorflow'

        if self.framework == 'tensorflow':
            self.explainer = TFDeepExplainer(model, data, session, learning_phase_flags)
        elif self.framework == 'pytorch':
            self.explainer = PyTorchDeepExplainer(model, data, feedforward_args)

        self.expected_value = self.explainer.expected_value

    def shap_values(self, X, ranked_outputs=None, output_rank_order='max', feedforward_args=None, var_seq_len=False, see_progress=False):
        """ Return approximate SHAP values for the model applied to the data given by X.

        Parameters
        ----------
        X : list,
            if framework == 'tensorflow': numpy.array, or pandas.DataFrame
            if framework == 'pytorch': torch.tensor
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        ranked_outputs : None or int
            If ranked_outputs is None then we explain all the outputs in a multi-output model. If
            ranked_outputs is a positive integer then we only explain that many of the top model
            outputs (where "top" is determined by output_rank_order). Note that this causes a pair
            of values to be returned (shap_values, indexes), where shap_values is a list of numpy
            arrays for each of the output ranks, and indexes is a matrix that indicates for each sample
            which output indexes were choses as "top".

        output_rank_order : "max", "min", or "max_abs"
            How to order the model outputs when using ranked_outputs, either by maximum, minimum, or
            maximum absolute value.

        if framework == 'pytorch':

        feedforward_args : None or list
            In case your model's feedforward method has additional parameters besides the input data,
            you can specify them in a list format in feedforward_args. This can be useful for instance
            if you're dealing with recurrent neural networks that receive inputs with variable sequence
            length, requiring padding and the list of original sequence lengths. Currently only works
            in PyTorch.

        var_seq_len : bool
            Indicates whether the input data has variable sequence length or not. If true, the lists of
            the original sequence lengths for the background data (used in the explainer) and of the
            test data (corresponding to X) must be provided as the first two items of feedforward_args.
            Currently only works in PyTorch.
            Usage example:
            explainer.shap_values(X, feedforward_args=[x_lenghts_background, x_lenghts_test], var_seq_len=True)

        see_progress : bool, default False
            If set to True, a progress bar will show up indicating the execution
            of the SHAP values calculations.

        Returns
        -------
        For a models with a single output this returns a tensor of SHAP values with the same shape
        as X. For a model with multiple outputs this returns a list of SHAP value tensors, each of
        which are the same shape as X. If ranked_outputs is None then this list of tensors matches
        the number of model outputs. If ranked_outputs is a positive integer a pair is returned
        (shap_values, indexes), where shap_values is a list of tensors with a length of
        ranked_outputs, and indexes is a matrix that indicates for each sample which output indexes
        were chosen as "top".
        """
        if self.framework == 'tensorflow':
            return self.explainer.shap_values(X, ranked_outputs, output_rank_order)
        elif self.framework == 'pytorch':
            return self.explainer.shap_values(X, ranked_outputs, output_rank_order, feedforward_args, var_seq_len, see_progress)
