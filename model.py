import pickle
from sklearn.neighbors import KNeighborsClassifier

class KNNUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        if name in ["EuclideanDistance64", "EuclideanDistance"] and module == "sklearn.metrics._dist_metrics":
            # Yerine None döndür — sonra metric string olarak ayarlanacak
            return lambda *args, **kwargs: None
        return super().find_class(module, name)

def fix_knn_pickle(old_path, new_path):
    with open(old_path, "rb") as f:
        model = KNNUnpickler(f).load()

    if isinstance(model, KNeighborsClassifier):
        model.metric = "euclidean"  # tüm sürümlerde çalışır
        model.effective_metric_ = "euclidean"
        model.effective_metric_params_ = {}

    with open(new_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Yeni KNN modeli kaydedildi: {new_path}")

# Kullanım
fix_knn_pickle("knn_model.pkl", "knn_model_universal.pkl")
