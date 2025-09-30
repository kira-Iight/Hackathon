import torch
from ultralytics import YOLO
import DetectionAndClassification as dac
import ClassificationTrain as ctrain

def main():
    porody_path = "data/species"
    char_path = "data/defects"  # Путь к данным дефектов
    
    try:
        detection_model = YOLO("best.pt")
    except Exception as e:
        print(f"{e}")
        return

    tree_model, bush_model, tree_class_names, bush_class_names = dac.load_separated_models()
    
    if tree_model is None and bush_model is None:

        tree_model, bush_model, tree_class_names, bush_class_names = ctrain.train_separated_models(porody_path)
        
        # Сохранение моделей пород
        if tree_model is not None:
            torch.save({
                'model_state_dict': tree_model.state_dict(),
                'class_names': tree_class_names,
                'plant_type': 'tree'
            }, 'model_trees.pth')
            print("Модель деревьев сохранена")
        
        if bush_model is not None:
            torch.save({
                'model_state_dict': bush_model.state_dict(),
                'class_names': bush_class_names, 
                'plant_type': 'bush'
            }, 'model_bushes.pth')
            print("Модель кустов сохранена")
    
    defects_model, defects_class_names = dac.load_defects_model()
    
    if defects_model is None:
        defects_model, defects_history, defects_class_names, _, _ = ctrain.train_defects_model_improved(char_path)
        
        if defects_model is not None:
            torch.save({
                'model_state_dict': defects_model.state_dict(),
                'class_names': defects_class_names,
                'model_type': 'defects'
            }, 'model_defects.pth')
            print("Модель дефектов сохранена")

#if __name__ == "__main__":
#    main()