"""Integration-specific runtime labels without device-name guessing."""

from dataclasses import replace

from measure.analyser.vacuum_signals import (
    ALIASES,
    Activity,
    discover_signals,
    resolve_activity,
    resolve_portable_entity,
)
from measure.recording.models import RecordedEntity, RecordedEntityState, RecordingContext, RecordingSample
import pytest

PRIMARY = "vacuum.robot"


def context(*entities: RecordedEntity) -> RecordingContext:
    return RecordingContext(
        "vacuum_robot",
        PRIMARY,
        "vacuum_robot",
        [
            RecordedEntity(PRIMARY, "vacuum", "primary", device_id="robot"),
            *entities,
        ],
    )


def entity(key: str, domain: str = "sensor", integration: str = "dreame_vacuum") -> RecordedEntity:
    return RecordedEntity(
        domain + "." + key, domain, "tracked", device_id="robot", translation_key=key, integration=integration
    )


def sample(state: str = "docked", **states: str) -> RecordingSample:
    return RecordingSample(
        0,
        5,
        {
            PRIMARY: RecordedEntityState(state, {}),
            **{key: RecordedEntityState(value, {}) for key, value in states.items()},
        },
    )


@pytest.mark.parametrize(
    "activity,label", [(activity, label) for activity, labels in ALIASES.items() for label in sorted(labels)]
)
def test_runtime_aliases(activity: Activity, label: str) -> None:
    item = sample(**{"sensor.status": label})
    assert resolve_activity(item, discover_signals([item], context(entity("status")))) is activity


@pytest.mark.parametrize(
    "state,activity",
    [
        ("cleaning", "away"),
        ("returning", "away"),
        ("idle", "away"),
        ("paused", "away"),
        ("docked", "docked"),
        ("error", None),
    ],
)
def test_standard_ha_activities(state: str, activity: str | None) -> None:
    item = sample(state)
    assert resolve_activity(item, discover_signals([item], context())) == activity


def test_dreame_state_wins_over_limited_charging_status_and_status() -> None:
    ctx = context(entity("charging_status"), entity("status"), entity("state"))
    items = [
        sample(**{"sensor.state": state, "sensor.charging_status": charging, "sensor.status": status})
        for state, charging, status in [
            ("washing", "not_charging", "standby"),
            ("charging", "charging", "charging"),
            ("charging_completed", "charging_completed", "standby"),
        ]
    ]
    signals = discover_signals(items, ctx)
    assert [resolve_activity(item, signals) for item in items] == ["washing", "charging", "completed"]
    assert {signal.feature.entity_id for signal in signals} == {"sensor.state"}


@pytest.mark.parametrize(
    "key,states,expected",
    [
        (
            "station_state",
            ["idle", "emptying_dustbin", "washing_mop", "drying_mop"],
            ["docked", "auto_emptying", "washing", "drying"],
        ),
        (
            "self_wash_base_status",
            ["idle", "washing", "drying", "clean_add_water", "adding_water", "returning", "paused"],
            ["docked", "washing", "drying", "washing", "washing", "docked", None],
        ),
        ("auto_empty_status", ["idle", "active", "not_performed"], ["docked", "auto_emptying", "docked"]),
        (
            "charging_status",
            ["not_charging", "charging", "charging_completed", "return_to_charge"],
            ["docked", "charging", "completed", "docked"],
        ),
    ],
)
def test_auxiliary_sensors(key: str, states: list[str], expected: list[str | None]) -> None:
    items = [sample(**{"sensor." + key: state}) for state in states]
    signals = discover_signals(items, context(entity(key)))
    assert [resolve_activity(item, signals) for item in items] == expected
    for state in ("unknown", "unavailable", "new_unmapped_state"):
        assert resolve_activity(sample(**{"sensor." + key: state}), signals) is None


def test_idle_station_preserves_cleaning_and_uses_one_guard() -> None:
    items = [sample(state, **{"sensor.station_state": "idle"}) for state in ("docked", "cleaning")]
    signals = discover_signals(items, context(entity("station_state", integration="ecovacs")))
    assert [resolve_activity(item, signals) for item in items] == ["docked", "away"]
    assert sum(signal.feature.entity_id == "sensor.station_state" for signal in signals) == 1
    assert not any(
        signal.feature.entity_id == "sensor.station_state"
        for signal in discover_signals(
            [sample(**{"sensor.station_state": "unknown"})], context(entity("station_state"))
        )
    )


def test_partial_charging_enum_supplements_completion_only() -> None:
    items = [
        sample(**{"sensor.state": state, "sensor.charging_status": charge})
        for state, charge in [("charging", "charging"), ("docked", "charging_completed"), ("cleaning", "not_charging")]
    ]
    signals = discover_signals(items, context(entity("state"), entity("charging_status")))
    assert [resolve_activity(item, signals) for item in items] == ["charging", "completed", "away"]


@pytest.mark.parametrize("attribute", [True, False])
def test_sleeping_supplements_completed(attribute: bool) -> None:
    ctx = context(entity("state"), entity("status"))
    items = [
        sample(**{"sensor.state": "charging_completed", "sensor.status": status}) for status in ("sleeping", "standby")
    ]
    if attribute:
        ctx = context(entity("state"))
        items = [
            replace(
                item,
                entities={
                    **item.entities,
                    PRIMARY: RecordedEntityState("docked", {"status": item.entities["sensor.status"].state}),
                },
            )
            for item in items
        ]
    signals = discover_signals(items, ctx)
    assert [resolve_activity(item, signals) for item in items] == ["sleeping", "completed"]


@pytest.mark.parametrize(
    "key,domain,integration,device_class,expected",
    [
        ("charging_state", "binary_sensor", "dreame_vacuum", None, ["charging", "docked"]),
        ("battery_charging", "binary_sensor", "ecovacs", "battery_charging", ["charging", "docked"]),
        ("battery_charging", "binary_sensor", "roborock", "battery_charging", ["docked", "docked"]),
        ("mop_drying", "switch", "roborock", None, ["drying", "docked"]),
        ("auto_drying", "switch", "dreame_vacuum", None, ["docked", "docked"]),
        ("water_mop_attached", "binary_sensor", "ecovacs", None, ["docked", "docked"]),
    ],
)
def test_runtime_flags_not_settings(
    key: str, domain: str, integration: str, device_class: str | None, expected: list[str]
) -> None:
    descriptor = replace(
        entity(key, domain, integration), device_class=device_class, translation_key=None if device_class else key
    )
    items = [sample(**{descriptor.entity_id: value}) for value in ("on", "off")]
    signals = discover_signals(items, context(descriptor))
    assert [resolve_activity(item, signals) for item in items] == expected


def test_unrelated_device_is_not_portable() -> None:
    descriptor = replace(entity("mop_drying", "switch", "roborock"), device_id="dock")
    ctx = context(descriptor)
    assert resolve_portable_entity(descriptor.entity_id, ctx) is None
    item = sample(**{descriptor.entity_id: "on"})
    assert resolve_activity(item, discover_signals([item], ctx)) == "docked"


def test_related_dock_entity_is_a_portable_activity_signal() -> None:
    drying = replace(entity("mop_drying", "switch", "roborock"), device_id="dock")
    ctx = replace(context(drying), related_device_ids=["dock"])
    assert resolve_portable_entity(drying.entity_id, ctx) == "[[entity_by_translation_key:mop_drying]]"

    items = [sample(**{drying.entity_id: value}) for value in ("on", "off")]
    signals = discover_signals(items, ctx)
    assert [resolve_activity(item, signals) for item in items] == ["drying", "docked"]
    assert signals[0].build_condition(ctx) == {
        "condition": "state",
        "entity_id": "[[entity_by_translation_key:mop_drying]]",
        "state": ["on"],
    }


@pytest.mark.parametrize(
    "other_device_id",
    [
        # PowerCalc resolves the key to the vacuum's own entity first.
        "robot",
        # PowerCalc refuses a key that matches on more than one related device.
        "second_dock",
    ],
)
def test_related_entity_needs_a_placeholder_resolving_to_it(other_device_id: str) -> None:
    drying = replace(entity("mop_drying", "switch", "roborock"), device_id="dock")
    other = replace(drying, entity_id="switch.other_mop_drying", device_id=other_device_id)
    ctx = replace(context(drying, other), related_device_ids=["dock", "second_dock"])
    assert resolve_portable_entity(drying.entity_id, ctx) is None


def test_related_battery_uses_device_class_when_the_vacuum_has_none() -> None:
    battery = RecordedEntity(
        "sensor.dock_battery", "sensor", "battery", device_class="battery", unit="%", device_id="dock"
    )
    ctx = replace(context(battery), related_device_ids=["dock"])
    assert resolve_portable_entity(battery.entity_id, ctx) == "[[entity_by_device_class:battery]]"


def test_disabled_sleep_status_is_ignored() -> None:
    items = [sample(**{"sensor.state": "charging_completed", "sensor.status": "sleeping"})]
    signals = discover_signals(items, context(entity("state"), replace(entity("status"), disabled_by="user")))
    assert resolve_activity(items[0], signals) == "completed"


def test_unavailable_action_sensor_does_not_override_standard_vacuum_activities() -> None:
    descriptor = entity("mop_drying", "switch", "roborock")
    items = [
        sample("docked", **{descriptor.entity_id: "unavailable"}),
        sample("cleaning", **{descriptor.entity_id: "unknown"}),
    ]

    signals = discover_signals(items, context(descriptor))

    assert all(signal.feature.entity_id != descriptor.entity_id for signal in signals)
    assert [resolve_activity(item, signals) for item in items] == [Activity.DOCKED, Activity.AWAY]


def test_numeric_action_attributes_are_not_interpreted_as_boolean_flags() -> None:
    items = [
        RecordingSample(
            index,
            5,
            {PRIMARY: RecordedEntityState(state, {"drying": value})},
        )
        for index, (state, value) in enumerate([("docked", 0), ("cleaning", 1)])
    ]

    signals = discover_signals(items, context())

    assert all(signal.feature.attribute != "drying" for signal in signals)
    assert [resolve_activity(item, signals) for item in items] == [Activity.DOCKED, Activity.AWAY]
