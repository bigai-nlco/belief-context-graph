"""Unit tests against a running Memgraph instance via the Bolt protocol.

Connects to the Memgraph container exposed at bolt://localhost:7687 (no auth)
and exercises CRUD plus a few transaction scenarios:

  - Create / Read / Update / Delete on nodes and relationships
  - Index + uniqueness constraint creation and enforcement
  - Explicit transaction commit & rollback
  - Managed transactions (execute_write / execute_read) with retry semantics
  - Money-transfer pattern: success commits both sides, failure rolls back both
  - Multi-statement order + items written atomically inside one transaction
  - Constraint violation rolls back every prior write in the same transaction

Run directly with `python scripts/test_memgraph.py` or via unittest:
    python -m unittest scripts.test_memgraph -v

Environment overrides:
    MEMGRAPH_URI   (default bolt://localhost:7687)
    MEMGRAPH_USER  (default empty)
    MEMGRAPH_PASS  (default empty)
"""

from __future__ import annotations

import os
import sys
import unittest
import uuid

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, ConstraintError, CypherSyntaxError


URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
USER = os.environ.get("MEMGRAPH_USER", "")
PASS = os.environ.get("MEMGRAPH_PASS", "")
LABEL_PREFIX = "MgTest"


def _label(name: str) -> str:
    return f"{LABEL_PREFIX}_{name}"


class MemgraphTestBase(unittest.TestCase):
    driver = None

    @classmethod
    def setUpClass(cls):
        cls.driver = GraphDatabase.driver(URI, auth=(USER, PASS))
        cls.driver.verify_connectivity()

    @classmethod
    def tearDownClass(cls):
        if cls.driver is not None:
            cls.driver.close()

    def setUp(self):
        with self.driver.session() as s:
            s.run(f"MATCH (n:{_label('Person')}) DETACH DELETE n")
            s.run(f"MATCH (n:{_label('Account')}) DETACH DELETE n")
            s.run(f"MATCH (n:{_label('Order')}) DETACH DELETE n")
            s.run(f"MATCH (n:{_label('Item')}) DETACH DELETE n")
            s.run(f"MATCH (n:{_label('Unique')}) DETACH DELETE n")
            for stmt in (
                f"DROP CONSTRAINT ON (a:{_label('Account')}) ASSERT a.id IS UNIQUE",
                f"DROP CONSTRAINT ON (u:{_label('Unique')}) ASSERT u.email IS UNIQUE",
                f"DROP INDEX ON :{_label('Person')}(name)",
            ):
                try:
                    s.run(stmt)
                except (ClientError, CypherSyntaxError):
                    pass

    def tearDown(self):
        self.setUp()


class CrudTests(MemgraphTestBase):
    def test_create_and_read_node(self):
        person = _label("Person")
        with self.driver.session() as s:
            s.run(
                f"CREATE (:{person} {{name: $name, age: $age}})",
                name="Alice",
                age=30,
            )
            rec = s.run(
                f"MATCH (p:{person} {{name: $name}}) RETURN p.age AS age",
                name="Alice",
            ).single()
            self.assertIsNotNone(rec)
            self.assertEqual(rec["age"], 30)

    def test_update_node_property(self):
        person = _label("Person")
        with self.driver.session() as s:
            s.run(f"CREATE (:{person} {{name: 'Bob', age: 25}})")
            s.run(
                f"MATCH (p:{person} {{name: 'Bob'}}) SET p.age = $age",
                age=26,
            )
            age = s.run(
                f"MATCH (p:{person} {{name: 'Bob'}}) RETURN p.age AS age"
            ).single()["age"]
            self.assertEqual(age, 26)

    def test_delete_node(self):
        person = _label("Person")
        with self.driver.session() as s:
            s.run(f"CREATE (:{person} {{name: 'Carol'}})")
            s.run(f"MATCH (p:{person} {{name: 'Carol'}}) DETACH DELETE p")
            count = s.run(
                f"MATCH (p:{person} {{name: 'Carol'}}) RETURN count(p) AS c"
            ).single()["c"]
            self.assertEqual(count, 0)

    def test_create_and_traverse_relationship(self):
        person = _label("Person")
        with self.driver.session() as s:
            s.run(
                f"CREATE (a:{person} {{name: 'Ann'}})-[:KNOWS {{since: 2020}}]->"
                f"(b:{person} {{name: 'Ben'}})"
            )
            rec = s.run(
                f"MATCH (a:{person} {{name: 'Ann'}})-[r:KNOWS]->(b:{person}) "
                "RETURN b.name AS name, r.since AS since"
            ).single()
            self.assertEqual(rec["name"], "Ben")
            self.assertEqual(rec["since"], 2020)

    def test_bulk_create_with_unwind(self):
        person = _label("Person")
        rows = [{"name": f"P{i}", "age": 20 + i} for i in range(5)]
        with self.driver.session() as s:
            s.run(
                f"UNWIND $rows AS row CREATE (:{person} {{name: row.name, age: row.age}})",
                rows=rows,
            )
            total = s.run(
                f"MATCH (p:{person}) RETURN count(p) AS c"
            ).single()["c"]
            self.assertEqual(total, 5)


class IndexAndConstraintTests(MemgraphTestBase):
    def test_label_property_index(self):
        person = _label("Person")
        with self.driver.session() as s:
            s.run(f"CREATE INDEX ON :{person}(name)")
            indexes = s.run("SHOW INDEX INFO").data()

            def matches(row) -> bool:
                if row.get("label") != person:
                    return False
                prop = row.get("property")
                if isinstance(prop, list):
                    return "name" in prop
                return prop == "name"

            self.assertTrue(
                any(matches(r) for r in indexes),
                f"index on :{person}(name) not reported by SHOW INDEX INFO",
            )

    def test_unique_constraint_blocks_duplicates(self):
        # Memgraph enforces unique constraints at commit time. Drive the
        # duplicate insert through an explicit transaction so commit() raises.
        unique = _label("Unique")
        with self.driver.session() as s:
            s.run(
                f"CREATE CONSTRAINT ON (u:{unique}) ASSERT u.email IS UNIQUE"
            )
            s.execute_write(
                lambda tx: tx.run(
                    f"CREATE (:{unique} {{email: 'a@example.com'}})"
                ).consume()
            )
            with self.assertRaises((ConstraintError, ClientError)):
                with s.begin_transaction() as tx:
                    tx.run(
                        f"CREATE (:{unique} {{email: 'a@example.com'}})"
                    ).consume()
                    tx.commit()
            count = s.run(
                f"MATCH (u:{unique} {{email: 'a@example.com'}}) RETURN count(u) AS c"
            ).single()["c"]
            self.assertEqual(count, 1)


class TransactionTests(MemgraphTestBase):
    def _create_account(self, tx, account_id: str, balance: int):
        account = _label("Account")
        tx.run(
            f"CREATE (:{account} {{id: $id, balance: $balance}})",
            id=account_id,
            balance=balance,
        )

    def _balance(self, session, account_id: str) -> int:
        account = _label("Account")
        rec = session.run(
            f"MATCH (a:{account} {{id: $id}}) RETURN a.balance AS b",
            id=account_id,
        ).single()
        return None if rec is None else rec["b"]

    def test_explicit_commit(self):
        account = _label("Account")
        with self.driver.session() as s:
            with s.begin_transaction() as tx:
                self._create_account(tx, "A1", 100)
                tx.commit()
            self.assertEqual(self._balance(s, "A1"), 100)

    def test_explicit_rollback(self):
        with self.driver.session() as s:
            with s.begin_transaction() as tx:
                self._create_account(tx, "A2", 50)
                tx.rollback()
            self.assertIsNone(self._balance(s, "A2"))

    def test_transfer_succeeds_atomically(self):
        account = _label("Account")
        with self.driver.session() as s:
            def seed(tx):
                self._create_account(tx, "src", 100)
                self._create_account(tx, "dst", 0)
            s.execute_write(seed)

            def transfer(tx, amount):
                tx.run(
                    f"MATCH (a:{account} {{id: 'src'}}) "
                    f"SET a.balance = a.balance - $amount",
                    amount=amount,
                )
                tx.run(
                    f"MATCH (b:{account} {{id: 'dst'}}) "
                    f"SET b.balance = b.balance + $amount",
                    amount=amount,
                )
                return tx.run(
                    f"MATCH (a:{account} {{id: 'src'}}), (b:{account} {{id: 'dst'}}) "
                    "RETURN a.balance AS src, b.balance AS dst"
                ).single().data()

            result = s.execute_write(transfer, 30)
            self.assertEqual(result, {"src": 70, "dst": 30})
            self.assertEqual(self._balance(s, "src"), 70)
            self.assertEqual(self._balance(s, "dst"), 30)

    def test_transfer_rolls_back_on_failure(self):
        """Mid-transaction Python exception must roll back every prior write."""
        account = _label("Account")
        with self.driver.session() as s:
            s.execute_write(lambda tx: self._create_account(tx, "src", 100))
            s.execute_write(lambda tx: self._create_account(tx, "dst", 0))

            class TransferAborted(RuntimeError):
                pass

            def faulty_transfer(tx):
                tx.run(
                    f"MATCH (a:{account} {{id: 'src'}}) "
                    "SET a.balance = a.balance - 40"
                )
                raise TransferAborted("simulated failure after first write")

            with self.assertRaises(TransferAborted):
                s.execute_write(faulty_transfer)

            self.assertEqual(self._balance(s, "src"), 100)
            self.assertEqual(self._balance(s, "dst"), 0)

    def test_multi_statement_order_is_atomic(self):
        order = _label("Order")
        item = _label("Item")
        order_id = f"ORD-{uuid.uuid4().hex[:8]}"
        with self.driver.session() as s:
            def place_order(tx):
                tx.run(
                    f"CREATE (:{order} {{id: $oid, total: 0}})",
                    oid=order_id,
                )
                for sku, price in [("SKU-1", 10), ("SKU-2", 25)]:
                    tx.run(
                        f"MATCH (o:{order} {{id: $oid}}) "
                        f"CREATE (o)-[:CONTAINS]->(:{item} {{sku: $sku, price: $price}}) "
                        "SET o.total = o.total + $price",
                        oid=order_id,
                        sku=sku,
                        price=price,
                    )
            s.execute_write(place_order)

            rec = s.run(
                f"MATCH (o:{order} {{id: $oid}})-[:CONTAINS]->(i:{item}) "
                "RETURN o.total AS total, count(i) AS items",
                oid=order_id,
            ).single()
            self.assertEqual(rec["total"], 35)
            self.assertEqual(rec["items"], 2)

    def test_constraint_violation_rolls_back_prior_writes(self):
        account = _label("Account")
        with self.driver.session() as s:
            s.run(
                f"CREATE CONSTRAINT ON (a:{account}) ASSERT a.id IS UNIQUE"
            )
            s.execute_write(lambda tx: self._create_account(tx, "X", 10))

            with self.driver.session() as s2:
                with self.assertRaises((ConstraintError, ClientError)):
                    with s2.begin_transaction() as tx:
                        self._create_account(tx, "Y", 99)
                        self._create_account(tx, "X", 0)
                        tx.commit()

            self.assertIsNone(self._balance(s, "Y"))
            self.assertEqual(self._balance(s, "X"), 10)


def _print_server_info():
    try:
        with GraphDatabase.driver(URI, auth=(USER, PASS)) as drv:
            with drv.session() as s:
                rec = s.run("SHOW VERSION").single()
                if rec:
                    print(f"# Memgraph server: {dict(rec)}", file=sys.stderr)
    except Exception:
        pass


if __name__ == "__main__":
    _print_server_info()
    unittest.main(verbosity=2)
