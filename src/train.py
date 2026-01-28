"""
모델 학습 스크립트
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dataset import download_esc50, ESC50Dataset
from model import FallDetectionCNN, FallDetectionResNet


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for mel_spec, labels in tqdm(dataloader, desc="Training"):
        mel_spec = mel_spec.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(mel_spec)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """모델 평가"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for mel_spec, labels in tqdm(dataloader, desc="Evaluating"):
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)

            outputs = model(mel_spec)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return avg_loss, accuracy, f1, precision, recall


def train(args):
    """메인 학습 함수"""
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 데이터셋 다운로드
    data_path = download_esc50(args.data_dir)

    # 데이터셋 로드
    train_dataset = ESC50Dataset(data_path, train=True, fold=args.fold)
    test_dataset = ESC50Dataset(data_path, train=False, fold=args.fold)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 모델 생성
    if args.model == "cnn":
        model = FallDetectionCNN()
    else:
        model = FallDetectionResNet()

    model = model.to(device)
    print(f"Model: {args.model}")

    # 손실 함수 및 옵티마이저
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 학습
    best_f1 = 0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 학습
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 평가
        test_loss, test_acc, f1, precision, recall = evaluate(
            model, test_loader, criterion, device
        )
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        scheduler.step(test_loss)

        # 베스트 모델 저장
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1,
                'accuracy': test_acc,
            }, save_path)
            print(f"Best model saved! F1: {f1:.4f}")

    print(f"\nTraining complete! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="낙상 감지 모델 학습")
    parser.add_argument("--data_dir", type=str, default="data", help="데이터 디렉토리")
    parser.add_argument("--save_dir", type=str, default="models", help="모델 저장 디렉토리")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fold", type=int, default=5, help="테스트용 fold (1-5)")

    args = parser.parse_args()
    train(args)
