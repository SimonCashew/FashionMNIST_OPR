import equinox as eqx
import jax
import jax.numpy as jnp
import optax  
import torch  
import torchvision  
from jaxtyping import Array, Float, Int, PyTree

import wandb
import hydra
from hydra import initialize, compose
import omegaconf
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append("..")
sys.path.append("../../orient/")

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(project="MNIST", config=cfg_container)
    cfg_dict = dict(wandb.config)
    cfg = OmegaConf.create(cfg_dict)

    LEARNING_RATE = cfg.lr_start
    REG_FACTOR = cfg.reg_factor
    STEPS = cfg.steps
    SEED = cfg.seed

    BATCH_SIZE = 64
    PRINT_EVERY = 30

    key = jax.random.PRNGKey(SEED)

    normalize_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = torchvision.datasets.FashionMNIST(
        "FashionMNIST",
        train=True,
        download=True,
        transform=normalize_data,
    )
    val_dataset = torchvision.datasets.FashionMNIST(
        "FashionMNIST",
        train=False,
        download=True,
        transform=normalize_data,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    class CNN(eqx.Module):
        layers: list

        def __init__(self, key):
            key1, key2, key3, key4 = jax.random.split(key, 4)
            self.layers = [
                eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
                eqx.nn.MaxPool2d(kernel_size=2),
                jax.nn.relu,
                jnp.ravel,
                eqx.nn.Linear(1728, 512, key=key2),
                jax.nn.sigmoid,
                eqx.nn.Linear(512, 64, key=key3),
                jax.nn.relu,
                eqx.nn.Linear(64, 10, key=key4),
                jax.nn.log_softmax,
            ]

        def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
            for layer in self.layers:
                x = layer(x)
            return x


    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey)

    def loss(model: CNN, x, y):
        pred_y = jax.vmap(model)(x)
        return cross_entropy(y, pred_y)


    def cross_entropy(
        y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
    ) -> Float[Array, ""]:
        pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
        return -jnp.mean(pred_y)


    loss = eqx.filter_jit(loss) 


    @eqx.filter_jit
    def compute_accuracy(
        model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
    ) -> Float[Array, ""]:
        pred_y = jax.vmap(model)(x)
        pred_y = jnp.argmax(pred_y, axis=1)
        return jnp.mean(y == pred_y)

    def evaluate(model: CNN, valloader: torch.utils.data.DataLoader):
        avg_loss = 0
        avg_acc = 0
        for x, y in valloader:
            x = x.numpy()
            y = y.numpy()
            avg_loss += loss(model, x, y)
            avg_acc += compute_accuracy(model, x, y)
        return avg_loss / len(valloader), avg_acc / len(valloader)

    evaluate(model, valloader)

    optim = optax.adamw(LEARNING_RATE)

    def train(
        model: CNN,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        optim: optax.GradientTransformation,
        steps: int,
        print_every: int,
    ) -> CNN:
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        
        @eqx.filter_jit
        def make_step(
            model: CNN,
            opt_state: PyTree,
            x: Float[Array, "batch 1 28 28"],
            y: Int[Array, " batch"],
        ):
            loss_value, grad = eqx.filter_value_and_grad(loss)(model, x, y)
            
            updates, opt_state = optim.update(
                grad, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        def infinite_trainloader():
            while True:
                yield from trainloader

        for step, (x, y) in zip(range(steps), infinite_trainloader()):
            x = x.numpy()
            y = y.numpy()
            model, opt_state, train_loss = make_step(model, opt_state, x, y)
            run.log({"loss": train_loss}, step=step)
            if (step % print_every) == 0 or (step == steps - 1):
                val_loss, val_accuracy = evaluate(model, valloader)
                run.log({"val_loss": val_loss}, step=step)
                run.log({"val_accuracy": val_accuracy}, step=step)
        return model

    model = train(model, trainloader, valloader, optim, STEPS, PRINT_EVERY)

    wandb.finish()

if __name__=='__main__':
    main()