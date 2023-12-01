import datetime
import enum
import os
import sqlite3
import uuid
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from contracts import NoneType
from utils.time import datetime_to_str, str_to_datetime

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

NULL_TOKEN = ":null"
VALUE_SEPARATOR = ", "
Value = Union[int, str, float, NoneType]


@dataclass
class EntryDataclass:
    @classmethod
    def get_metatype_mapping(cls) -> Dict[str, Type]:
        return {field_value.name: field_value.type for field_value in fields(cls)}

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Value]) -> "EntryDataclass":
        metatype_mapping = cls.get_metatype_mapping()
        typed_dict = {}
        for field_name, field_value in json_dict.items():
            field_type = metatype_mapping[field_name]
            if str(field_type).startswith("typing.Optional["):
                field_type = field_type.__args__[0]

            if field_value == NULL_TOKEN:
                typed_dict[field_name] = None
            elif field_type == datetime.datetime:
                typed_dict[field_name] = str_to_datetime(field_value)
            else:
                typed_dict[field_name] = field_type(field_value)

        return cls(**typed_dict)  # type: ignore

    def to_json_dict(self) -> Dict[str, Value]:
        metatype_mapping = self.get_metatype_mapping()
        json_dict = {}
        for field_name, field_value in vars(self).items():
            if field_value is None:
                continue

            field_type = metatype_mapping[field_name]
            if str(field_type).startswith("typing.Optional["):
                field_type = field_type.__args__[0]

            if field_type == datetime.datetime:
                json_dict[field_name] = datetime_to_str(field_value)
            elif isinstance(field_type, enum.EnumMeta):
                json_dict[field_name] = str(field_value.value)
            else:
                json_dict[field_name] = str(field_value)

        return json_dict


class Table:
    HIDDEN_COLUMNS: Tuple[str, ...] = ("row_uuid", "created_on", "last_modified")

    def __init__(self, name: str, entry_dataclass: Type[EntryDataclass]):
        self.name = name
        self.entry_dataclass = entry_dataclass
        self.columns_metatype = self.entry_dataclass.get_metatype_mapping()
        assert len(self.columns_metatype) > 0
        intersection = set(self.columns_metatype.keys()).intersection(
            self.HIDDEN_COLUMNS
        )
        assert intersection == set(), self
        self.columns = self.HIDDEN_COLUMNS + tuple(sorted(self.columns_metatype.keys()))


@dataclass
class TableEntry:
    row_uuid: uuid.UUID
    created_on: datetime.datetime
    last_modified: datetime.datetime
    entry: EntryDataclass

    @staticmethod
    def from_entry(entry: EntryDataclass) -> "TableEntry":
        return TableEntry(
            row_uuid=uuid.uuid4(),
            created_on=datetime.datetime.now(),
            last_modified=datetime.datetime.now(),
            entry=entry,
        )

    def to_str(self, table: Table) -> str:
        assert table.columns[:3] == Table.HIDDEN_COLUMNS, table.columns
        json_dict = self.to_json_dict()

        formatted_dic = {
            "row_uuid": str(self.row_uuid),
            "created_on": datetime_to_str(self.created_on),
            "last_modified": datetime_to_str(self.last_modified),
        }

        formatted_dic.update(json_dict["entry"])
        values = []
        for column in table.columns:
            value = f"'{str(formatted_dic.get(column) or NULL_TOKEN)}'"
            assert "," not in value
            values.append(value)

        row_str = VALUE_SEPARATOR.join(values).strip(VALUE_SEPARATOR)
        return f"({row_str})"

    def to_json_dict(self, flatten: bool = False) -> Dict[str, Any]:
        json_dict = {
            "row_uuid": str(self.row_uuid),
            "created_on": datetime_to_str(self.created_on),
            "last_modified": datetime_to_str(self.created_on),
        }
        if flatten:
            json_dict.update(self.entry.to_json_dict())
        else:
            json_dict["entry"] = self.entry.to_json_dict()

        return json_dict

    @staticmethod
    def from_json_dict(
        json_dict: Dict[str, Value], entry_dataclass: Type[EntryDataclass]
    ) -> "TableEntry":
        row_uuid = uuid.UUID(json_dict.get("row_uuid") or str(uuid.uuid4()))
        created_on = str_to_datetime(
            json_dict.get("created_on") or str(datetime.datetime.now())
        )
        last_modified = str_to_datetime(
            json_dict.get("last_modified") or str(datetime.datetime.now())
        )
        del json_dict["row_uuid"]
        del json_dict["created_on"]
        del json_dict["last_modified"]
        entry = entry_dataclass.from_json_dict(json_dict)
        return TableEntry(row_uuid, created_on, last_modified, entry)


class SqliteWrapper:
    def __init__(self, database_path: str, fail_if_does_not_exists: bool = True):
        self.fail_if_does_not_exists = fail_if_does_not_exists
        self.database_path = database_path

        if os.path.exists(self.database_path):
            self.connection = sqlite3.connect(self.database_path)
        elif self.fail_if_does_not_exists:
            raise FileNotFoundError(f"database_path = {self.database_path}")
        else:
            self.connection = self.init_database()

        self.cursor: sqlite3.Cursor = self.connection.cursor()

    def init_database(self) -> sqlite3.Connection:
        assert not os.path.exists(self.database_path)
        connection = sqlite3.connect(self.database_path)
        return connection

    def list_tables(self) -> Tuple[str, ...]:
        res = self.cursor.execute("SELECT name FROM sqlite_master")
        return sum(res.fetchall() or (), ())

    def create_table(self, table: Table) -> None:
        existing_tables = self.list_tables()
        assert table.name not in existing_tables, (table.name, existing_tables)
        columns_str = ", ".join(table.columns)
        self.cursor.execute(f"CREATE TABLE {table.name}({columns_str})")
        self.connection.commit()
        assert self.table_exists(table.name)

    def table_exists(self, table_name: str) -> bool:
        res = self.cursor.execute(
            f"SELECT name FROM sqlite_master WHERE name='{table_name}'"
        )
        tables = res.fetchone()
        return tables is not None and table_name in tables

    def table_columns(self, table_name: str) -> Tuple[str, ...]:
        assert self.table_exists(
            table_name
        ), f"Table {table_name} does not exist in {self.database_path}"
        res = self.cursor.execute(f"select * from {table_name}")
        columns = tuple(description[0] for description in res.description or ())
        return columns

    def verify_table(self, table: Table) -> None:
        columns = self.table_columns(table.name)
        assert columns == table.columns, (columns, table.columns)

    def add_entries(self, table: Table, entries: List[TableEntry]) -> None:
        datetime_now = datetime.datetime.now()
        for entry in entries:
            entry.created_on = datetime_now
            entry.last_modified = datetime_now

        entries_values = ",\n".join([entry.to_str(table) for entry in entries])
        self.cursor.execute(
            f"""
            INSERT INTO {table.name} VALUES 
            {entries_values}
            """,
            {NULL_TOKEN[1:]: None},
        )
        self.connection.commit()

    def add_entry(self, table: Table, entry: TableEntry) -> None:
        self.add_entries(table, [entry])

    def update_entry(self, table: Table, entry: TableEntry) -> None:
        entries = self.read_table(table)
        row_uuids = {str(entry.row_uuid) for entry in entries}
        assert str(entry.row_uuid) in row_uuids

        datetime_now = datetime.datetime.now()
        entry.last_modified = datetime_now
        new_values = ", ".join(
            f"{column} = '{value}'"
            for column, value in entry.to_json_dict(flatten=True).items()
            if column not in ["row_uuid", "created_on"]
        )
        self.cursor.execute(
            f"""
            UPDATE {table.name}
            SET {new_values}
            WHERE row_uuid='{entry.row_uuid}'
            """,
            {NULL_TOKEN[1:]: None},
        )
        self.connection.commit()

    def read_table(
        self,
        table: Table,
        columns: Tuple[str] = ("*",),
    ) -> List[TableEntry]:
        columns_names = self.table_columns(table.name)
        res = self.cursor.execute(
            f"""
            SELECT {' '.join(columns)}
            FROM {table.name}
            """,
            {NULL_TOKEN[1:]: None},
        )
        rows = res.fetchall()
        json_dicts = [dict(zip(columns_names, row)) for row in rows]
        return [
            TableEntry.from_json_dict(json_dict, table.entry_dataclass)
            for json_dict in json_dicts
        ]

    def find_entry(
        self,
        table: Table,
        uuid_key: str,
        uuid_value: uuid.UUID,
    ) -> Optional[TableEntry]:
        entries = self.read_table(table)
        matching_entries = [
            table_entry
            for table_entry in entries
            if getattr(table_entry.entry, uuid_key) == uuid_value
        ]
        assert (
            len(matching_entries) <= 1
        ), f"Multiple entries for {uuid_key} = {uuid_value} in table {table.name}"

        entry = None
        if matching_entries:
            entry = matching_entries[0]

        return entry

    def delete_entry(
        self,
        table: Table,
        uuid_key: str,
        uuid_value: uuid.UUID,
    ) -> None:
        entry = self.find_entry(table, uuid_key, uuid_value)
        if entry is None:
            assert False, f"Nothing to delete {table.name} {uuid_key} {uuid_value}"

        self.cursor.execute(
            f"""
            DELETE 
            FROM {table.name}
            WHERE {uuid_key}='{uuid_value}'
            """
        )
        self.connection.commit()
