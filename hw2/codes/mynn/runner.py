import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=128, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.batch_size = batch_size
        self.scheduler = scheduler

        # 必须初始化！
        self.train_loss = []
        self.train_scores = []
        self.dev_loss = []
        self.dev_scores = []
        self.best_score = 0

    def _clear_grads(self):
        for layer in self.model.layers:
            if hasattr(layer, 'clear_grad'):
                layer.clear_grad()

    def train(self, train_set, dev_set, **kwargs):
        self.early_stop_patience = kwargs.get("early_stop_patience", None)
        self.no_improve_count = 0
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 50)
        save_dir = kwargs.get("save_dir", "./best_models")
        
        # ✅ 新增：自定义保存文件名（默认 bes
        # t_model，但可以改）
        save_name = kwargs.get("save_name", "best_model.pickle")

        os.makedirs(save_dir, exist_ok=True)

        X_train, y_train = train_set
        X_dev, y_dev = dev_set
        global_step = 0

        for epoch in range(num_epochs):
            idx = np.random.permutation(len(X_train))
            X_train = X_train[idx]
            y_train = y_train[idx]
            total_iters = len(X_train) // self.batch_size

            pbar = tqdm(range(total_iters), desc=f"Epoch {epoch+1}/{num_epochs}")
            for iteration in pbar:
                start = iteration * self.batch_size
                end = start + self.batch_size
                batch_X = X_train[start:end]
                batch_y = y_train[start:end]

                self._clear_grads()
                logits = self.model(batch_X)
                loss = self.loss_fn(logits, batch_y)
                self.loss_fn.backward()
                self.optimizer.step()

                train_acc = self.metric(logits, batch_y)
                self.train_loss.append(loss)
                self.train_scores.append(train_acc)

                pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{train_acc:.4f}"})

                # ✅ 【关键修改】仅加这3行，不影响其他实验
                if self.scheduler is not None:
                    self.scheduler.step()

                if global_step % log_iters == 0:
                    dev_logits = self.model(X_dev)
                    dev_loss = self.loss_fn(dev_logits, y_dev)
                    dev_acc = self.metric(dev_logits, y_dev)
                    self.dev_loss.append(dev_loss)
                    self.dev_scores.append(dev_acc)
                    pbar.write(f"\n[Train] loss: {loss:.4f}, acc: {train_acc:.4f}")
                    pbar.write(f"[Dev] loss: {dev_loss:.4f}, acc: {dev_acc:.4f}")
                    
                    if dev_acc > self.best_score:
                        self.best_score = dev_acc
                        self.save_model(os.path.join(save_dir, save_name))
                        pbar.write(f"✅ New best dev acc: {self.best_score:.4f}")
                        self.no_improve_count = 0
                    else:
                        self.no_improve_count += 1
                        if self.early_stop_patience is not None and self.no_improve_count >= self.early_stop_patience:
                            pbar.write(f"\n⏹️ Early stopping triggered after {global_step} steps")
                            return

                global_step += 1

    def save_model(self, save_path):
        import pickle
        params_to_save = []
        for layer in self.model.layers:
            if layer.optimizable:
                param_dict = {
                    'W': layer.W.copy(),
                    'b': layer.b.copy(),
                    'weight_decay_lambda': layer.weight_decay_lambda if hasattr(layer, 'weight_decay_lambda') else 0.0
                }
                params_to_save.append(param_dict)
        
        with open(save_path, "wb") as f:
            pickle.dump(params_to_save, f)