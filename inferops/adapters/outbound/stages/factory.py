"""Project stage construction at the outbound adapter boundary."""
from inferops.adapters.outbound.stages.registry import BUILTINS


def stage_factory(definition):
    def build(declaration):
        target = BUILTINS.get(declaration.stage, declaration.stage)
        return definition.object_loader(target)(**declaration.params)
    return build
