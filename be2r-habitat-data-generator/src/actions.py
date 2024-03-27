import attr
import magnum as mn
import numpy as np

import habitat_sim


@attr.s(auto_attribs=True, slots=True)
class FreqActuationSpec:
    amount: float
    multiplier: int


@habitat_sim.registry.register_move_fn(body_action=True)
class MoveForwardFreq(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: FreqActuationSpec
    ):
        forward_ax = (
            np.array(scene_node.absolute_transformation().rotation_scaling())
            @ habitat_sim.geo.FRONT
        )
        scene_node.translate_local(forward_ax * actuation_spec.amount / actuation_spec.multiplier)


@habitat_sim.registry.register_move_fn(body_action=True)
class TurnLeftFreq(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: FreqActuationSpec
    ):
        # Rotate about the +y (up) axis
        rotation_ax = habitat_sim.geo.UP
        scene_node.rotate_local(mn.Deg(actuation_spec.amount / actuation_spec.multiplier), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()


@habitat_sim.registry.register_move_fn(body_action=True)
class TurnRightFreq(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: FreqActuationSpec
    ):
        # Rotate about the +y (up) axis
        rotation_ax = habitat_sim.geo.UP
        scene_node.rotate_local(mn.Deg(-actuation_spec.amount / actuation_spec.multiplier), rotation_ax)
        # Calling normalize is needed after rotating to deal with machine precision errors
        scene_node.rotation = scene_node.rotation.normalized()