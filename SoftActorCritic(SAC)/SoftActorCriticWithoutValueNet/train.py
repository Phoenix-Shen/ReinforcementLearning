# %%
import yaml
with open("./SoftActorCritic(SAC)/SoftActorCriticWithoutValueNet/settings.yaml", "r", encoding="utf-8") as f:
    args = yaml.load(f.read(), Loader=yaml.FullLoader)

print(type(args))
# %%
for key, val in args.items():
    print(type(val), val)
