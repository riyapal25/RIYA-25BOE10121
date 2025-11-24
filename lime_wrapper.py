# src/explainability/lime_wrapper.py
from lime.lime_text import LimeTextExplainer

def explain_with_lime(model, text, class_names=None, num_features=6):
    """
    model: pipeline that has predict_proba and accepts raw text
    text: raw text string
    class_names: list of class names for interpreter
    """
    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(text, model.predict_proba, num_features=num_features)
    return {
        'local_exp': explanation.as_list(),
        'raw': explanation.as_map()
    }
