
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import copy
from model import get_model, SimpleMLP
from dataset import get_data
from utils import setup_logger, set_seed, format_confusion_matrix
from scheduler import WarmupCosineLR
import os
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                else:
                    outputs = outputs[0]
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    metrics = {
        'loss': total_loss / len(data_loader), 'accuracy': 100 * correct / total,
        'weighted_f1': report['weighted avg']['f1-score'],
        'classification_report': report, 'confusion_matrix': cm
    }
    return metrics

# def run_single_training(args, logger):
#     logger.info(f"--- Running Single Training Experiment ---")
#     set_seed(args.seed)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Using device: {device}")

#     g = torch.Generator().manual_seed(args.seed)
    
#     resize_for_inception = (args.model == 'inception_v3')
#     train_loader, test_loader, metadata = get_data(
#         args.dataset, args.batch_size, args.seed, g,transfer_learning=args.transfer_learning,daset_percentage=args.ds_percentage, noise_level=args.noise)

    
#     model = get_model(
#         model_name=args.model, num_classes=metadata['num_classes'],
#         transfer_learning=args.transfer_learning, freeze_layers=not args.fine_tune, classifier=args.classifier
#     ).to(device)
#     logger.info(f"Model '{args.model}' initialized. Pre-trained: {args.transfer_learning}. Layers frozen: {not args.fine_tune}")

#     criterion = nn.CrossEntropyLoss()
#     # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
#     optimizer = torch.optim.SGD(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=args.learning_rate,
#         weight_decay=args.weight_decay,
#         momentum=0.9,
#         nesterov=True,
#     )

#     # total_steps = args.num_epochs * len(train_loader)
#     # warmup_steps = int(total_steps * 0.3) # 30% of training is for warmup
#     # scheduler = WarmupCosineLR(optimizer, warmup_epochs=warmup_steps, max_epochs=total_steps)
#     # logger.info(f"Optimizer: SGD. Scheduler: WarmupCosineLR with {warmup_steps} warmup steps out of {total_steps} total steps.")

#     for epoch in range(args.num_epochs):
#         model.train()
#         for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
#             data, labels = data.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(data)
#             if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
#                 if hasattr(outputs, 'logits'):
#                     outputs = outputs.logits
#                 else:
#                     outputs = outputs[0]
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             # scheduler.step()
        
#         val_metrics = evaluate(model, test_loader, criterion, device)

#         # Extract classifier layer
#         classifier_layer = None
#         if hasattr(model, 'fc'):
#             classifier_layer = model.fc
#         elif hasattr(model, 'classifier'):
#             if isinstance(model.classifier, nn.Sequential):
#                 classifier_layer = model.classifier[-1]
#             else:
#                 classifier_layer = model.classifier

#         print(classifier_layer.__class__.__name__)
#         if classifier_layer is not None and classifier_layer.__class__.__name__ == 'GNUNet':
#             alphas = classifier_layer.get_alphas()
#             logger.info(f"Layer alphas: {alphas}")
#         # logger.info(f"Epoch [{epoch+1}/{args.num_epochs}] | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | Val F1: {val_metrics['weighted_f1']:.4f}")
#         current_lr = optimizer.param_groups[0]['lr']
#         logger.info(
#             f"Epoch [{epoch+1}/{args.num_epochs}] | Val Loss: {val_metrics['loss']:.4f} | "
#             f"Val Acc: {val_metrics['accuracy']:.2f}% | Val F1: {val_metrics['weighted_f1']:.4f} | LR: {current_lr:.6f} | "
#         )
#     logger.info("--- Final Evaluation Report ---")
#     final_metrics = evaluate(model, test_loader, criterion, device)

#     logger.info(
#     f"Classification Report: \n"
#     f"loss: {final_metrics['loss']}, "
#     f"accuracy: {final_metrics['accuracy']}, "
#     f"weighted_f1: {final_metrics['weighted_f1']}"
# )
#     # report_str = classification_report(np.arange(metadata['num_classes']), np.arange(metadata['num_classes']), labels=range(metadata['num_classes']), target_names=metadata['class_names'], output_dict=True, zero_division=0)
#     # logger.info("Classification Report:\n" + classification_report(final_metrics['classification_report']['accuracy'], final_metrics['classification_report']['accuracy'], labels=range(len(metadata['class_names'])), target_names=metadata['class_names'], digits=4, zero_division=0))
#     # logger.info("Confusion Matrix:" + format_confusion_matrix(final_metrics['confusion_matrix'], metadata['class_names']))



def run_single_training(args, logger):
    logger.info(f"--- Running Single Training Experiment ---")
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    g = torch.Generator().manual_seed(args.seed)
    
    resize_for_inception = (args.model == 'inception_v3')
    train_loader, test_loader, metadata = get_data(
        args.dataset, args.batch_size, args.seed, g, transfer_learning=args.transfer_learning,
        daset_percentage=args.ds_percentage, noise_level=args.noise
    )
    
    model = get_model(
        model_name=args.model, num_classes=metadata['num_classes'],
        transfer_learning=args.transfer_learning, freeze_layers=not args.fine_tune, classifier=args.classifier,alpha1=args.alpha1, alpha2=args.alpha2,projection=args.projection,fixed_alpha=args.fixed_alpha
    ).to(device)
    logger.info(f"Model '{args.model}' initialized. Pre-trained: {args.transfer_learning}. Layers frozen: {not args.fine_tune}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=0.9,
        nesterov=True,
    )
    
    # <<< START: ADDED FOR EARLY STOPPING >>>
    # Initialize variables for early stopping
    patience = args.patience
    epochs_no_improve = 0
    best_val_loss = float('inf')
    # Use deepcopy to store the best model state in memory
    best_model_weights = copy.deepcopy(model.state_dict())
    logger.info(f"Early stopping enabled with patience of {patience} epochs.")
    # <<< END: ADDED FOR EARLY STOPPING >>>

    for epoch in range(args.num_epochs):
        model.train()
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            if isinstance(outputs, tuple) or hasattr(outputs, 'logits'):
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                else:
                    outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # --- Validation Step ---
        val_metrics = evaluate(model, test_loader, criterion, device)
        current_val_loss = val_metrics['loss']
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch [{epoch+1}/{args.num_epochs}] | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}% | Val F1: {val_metrics['weighted_f1']:.4f} | LR: {current_lr:.6f}"
        )

        # Extract classifier layer (your existing logic)
        classifier_layer = None
        if hasattr(model, 'fc'):
            classifier_layer = model.fc
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                classifier_layer = model.classifier[-1]
            else:
                classifier_layer = model.classifier

        if classifier_layer is not None and hasattr(classifier_layer, 'get_alphas'):
            alphas = classifier_layer.get_alphas()
            logger.info(f"Layer alphas: {alphas}")
        
        # <<< START: ADDED FOR EARLY STOPPING >>>
        # Early stopping logic: check if validation loss has improved
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0
            # Save the best model weights
            best_model_weights = copy.deepcopy(model.state_dict())
            logger.info(f"Validation loss improved to {best_val_loss:.4f}. Saving model.")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        # If validation loss has not improved for 'patience' epochs, stop training
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
            break
        # <<< END: ADDED FOR EARLY STOPPING >>>

    # <<< START: ADDED FOR EARLY STOPPING >>>
    # Load the best model weights for the final evaluation
    logger.info(f"Training finished. Loading best model with val_loss: {best_val_loss:.4f} for final evaluation.")
    model.load_state_dict(best_model_weights)
    # <<< END: ADDED FOR EARLY STOPPING >>>

    logger.info("--- Final Evaluation Report ---")
    final_metrics = evaluate(model, test_loader, criterion, device)

    logger.info(
        f"Classification Report: \n"
        f"loss: {final_metrics['loss']:.4f}, "
        f"accuracy: {final_metrics['accuracy']:.2f}%, "
        f"weighted_f1: {final_metrics['weighted_f1']:.4f}"
    )

def run_hyperparameter_sweep(args, logger):
    logger.info(f"--- Running Hyperparameter Tuning for MLP on {args.dataset} ---")
    all_results = []
    for dim in tqdm(args.hidden_dims, desc="Testing Hidden Dimensions"):
        logger.info(f"--- Testing Hidden Dimension: {dim} ---")
        set_seed(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = torch.Generator().manual_seed(args.seed)
        resize_for_inception = (args.model == 'inception_v3')
        train_loader, test_loader, metadata = get_data( args.dataset, args.batch_size, args.seed, g,resize_for_inception=resize_for_inception)
        #train_loader, test_loader, metadata = get_data(args.dataset, args.batch_size, args.seed, g)
        model = SimpleMLP(metadata['input_size'], dim, metadata['num_classes']).to(device)
        criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        for epoch in range(args.num_epochs):
            model.train()
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad(); criterion(model(data), labels).backward(); optimizer.step()
        
        final_metrics = evaluate(model, test_loader, criterion, device)
        all_results.append({'hidden_dim': dim, **final_metrics})

    logger.info("\n--- Experiment Summary ---")
    logger.info(f"{'Hidden Dim':<12} | {'Accuracy (%)':<15} | {'Weighted F1-Score':<20} | {'Loss':<10}")
    logger.info(f"{'-'*12} | {'-'*15} | {'-'*20} | {'-'*10}")
    for res in all_results:
        logger.info(f"{res['hidden_dim']:<12} | {res['accuracy']:<15.2f} | {res['weighted_f1']:<20.4f} | {res['loss']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Classification Template")
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'tune'], help='Execution mode: single experiment or hyperparameter tuning.')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['iris', 'mnist', 'cifar10','cifar100','stl10','tiny_imagenet'])
    parser.add_argument('--model', type=str, default='resnet18', choices=['mlp', 'simple_cnn', 'resnet18','resnet34', 'resnet50', 'vgg11', 'vgg16', 'vgg19','densenet121', 'densenet169', 'densenet201', 'densenet161', 'mobilenet_v2', 'googlenet'], help='Model architecture to use.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-transfer-learning', action='store_false', dest='transfer_learning', help="Disable transfer learning.")
    parser.add_argument('--fine-tune', action='store_false', help="Enable fine-tuning of pre-trained layers.")
    # parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128], help='List of hidden dimensions for MLP tuning.')
    parser.add_argument('--classifier', type=str, default='mlp', choices=['linear', 'mlp', 'kan', 'gnu'], help='Classifier type for MLP-KAN.')
    parser.add_argument('--alpha1', type=float, default=0.5, help='Alpha value for MLP-KAN classifier.')
    parser.add_argument('--alpha2', type=float, default=0.5, help='Alpha value for MLP-KAN classifier.')
    parser.add_argument('--activation', type=str, default='identity', choices=['relu', 'silu','identity'], help='Activation function for MLP-KAN classifier.')
    parser.add_argument('--ds_percentage', type=float, default=1.0, help='Percentage of dataset to use for training (0.0 to 1.0).')
    parser.add_argument('--grid', type=int, default=5, help='Grid size for MLP-KAN classifier.')
    parser.add_argument('--degree', type=int, default=3, help='Degree for polynomial features in MLP-KAN classifier.')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level for datasets.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    # parser.add_argument()
    parser.add_argument('--projection', action='store_true', help="Enable projection layers.")
    # ablation study parameters
    parser.add_argument('--fixed_alpha', action='store_true', help="Fix alpha values during training.")
    args = parser.parse_args()
    
    folder_name = 'logs'
    os.makedirs(folder_name, exist_ok=True)
    if args.projection: 
        log_file = f"{folder_name}/Dataset_{args.dataset}_percentage_{args.ds_percentage}_noise_{args.noise}_model_{args.model}_classifier_{args.classifier}_activation_{args.activation}_degree_{args.degree}_grid_{args.grid}_alp1_{args.alpha1}_alp2_{args.alpha2}_epoch_{args.num_epochs}_projection_linear_seed_{args.seed}.log"
        # log_file = f"{args.dataset}_{args.model if args.mode == 'single' else 'mlp-tune'}_{args.classifier}.log"
    elif args.fixed_alpha:
        log_file = f"{folder_name}/Dataset_{args.dataset}_percentage_{args.ds_percentage}_noise_{args.noise}_model_{args.model}_classifier_{args.classifier}_activation_{args.activation}_degree_{args.degree}_grid_{args.grid}_alp1_{args.alpha1}_alp2_{args.alpha2}_epoch_{args.num_epochs}_fixed_alpha_seed_{args.seed}.log"
    else:
        log_file = f"{folder_name}/Dataset_{args.dataset}_percentage_{args.ds_percentage}_noise_{args.noise}_model_{args.model}_classifier_{args.classifier}_activation_{args.activation}_degree_{args.degree}_grid_{args.grid}_alp1_{args.alpha1}_alp2_{args.alpha2}_epoch_{args.num_epochs}_seed_{args.seed}.log"
    logger = setup_logger(log_file)
    logger.info(f"Arguments: {vars(args)}")

    if args.mode == 'single': run_single_training(args, logger)
    elif args.mode == 'tune': run_hyperparameter_sweep(args, logger)
