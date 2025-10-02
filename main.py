#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Powered Billing System (CLI + optional Tkinter GUI)
- Enter items (name, price, quantity)
- Computes subtotal, tax, discount, and grand total
- Saves invoice to CSV and pretty-printed TXT
- Optional simple ML model (scikit-learn) to predict likelihood of discount need

Beginner-friendly: run `python main.py` for CLI, or `python main.py --gui` for GUI.
"""

import argparse
import csv
import datetime as dt
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

# Optional ML imports guarded at runtime
try:
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

DEFAULT_TAX_RATE = 0.10  # 10%
DEFAULT_DISCOUNT_RATE = 0.05  # 5% discount if applied
INVOICES_DIR = "invoices"

@dataclass
class Item:
    name: str
    price: float
    quantity: int

@dataclass
class Invoice:
    items: List[Item]
    tax_rate: float = DEFAULT_TAX_RATE
    discount_rate: float = 0.0  # Applied discount rate (0 if none)

    @property
    def subtotal(self) -> float:
        return sum(i.price * i.quantity for i in self.items)

    @property
    def tax(self) -> float:
        return self.subtotal * self.tax_rate

    @property
    def discount_amount(self) -> float:
        return self.subtotal * self.discount_rate

    @property
    def total(self) -> float:
        return self.subtotal + self.tax - self.discount_amount

    def to_rows(self):
        for i in self.items:
            yield [i.name, f"{i.price:.2f}", i.quantity, f"{i.price * i.quantity:.2f}"]


def ensure_invoices_dir():
    if not os.path.exists(INVOICES_DIR):
        os.makedirs(INVOICES_DIR, exist_ok=True)


def save_invoice_csv_txt(inv: Invoice, buyer: str = "Customer") -> str:
    ensure_invoices_dir()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"invoice_{timestamp}"
    csv_path = os.path.join(INVOICES_DIR, base + ".csv")
    txt_path = os.path.join(INVOICES_DIR, base + ".txt")

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Buyer", buyer])
        writer.writerow(["Date", dt.datetime.now().isoformat(timespec="seconds")])
        writer.writerow([])
        writer.writerow(["Item", "Price", "Qty", "Line Total"])
        for row in inv.to_rows():
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["Subtotal", f"{inv.subtotal:.2f}"])
        writer.writerow(["Tax", f"{inv.tax:.2f}"])
        writer.writerow(["Discount", f"-{inv.discount_amount:.2f}"])
        writer.writerow(["Total", f"{inv.total:.2f}"])

    # TXT (pretty)
    lines = []
    lines.append("=" * 40)
    lines.append("AI Billing System - Invoice")
    lines.append("=" * 40)
    lines.append(f"Buyer: {buyer}")
    lines.append(f"Date: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("-" * 40)
    lines.append(f"{'Item':20} {'Price':>8} {'Qty':>5} {'Total':>8}")
    for it in inv.items:
        lines.append(f"{it.name:20} {it.price:>8.2f} {it.quantity:>5d} {it.price*it.quantity:>8.2f}")
    lines.append("-" * 40)
    lines.append(f"Subtotal: {inv.subtotal:.2f}")
    lines.append(f"Tax @ {inv.tax_rate*100:.1f}%: {inv.tax:.2f}")
    if inv.discount_rate > 0:
        lines.append(f"Discount @ {inv.discount_rate*100:.1f}%: -{inv.discount_amount:.2f}")
    lines.append(f"TOTAL: {inv.total:.2f}")
    lines.append("=" * 40)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return txt_path


class SimpleDiscountModel:
    """A minimal ML model that predicts if a customer may need a discount.

    Features (very simple, demo only):
    - subtotal
    - number of items
    Returns probability of needing discount. If > 0.5, apply DEFAULT_DISCOUNT_RATE.
    """

    def __init__(self):
        self.model: Optional[LogisticRegression] = None if SKLEARN_AVAILABLE else None

    def fit_demo(self):
        if not SKLEARN_AVAILABLE:
            return
        # Generate synthetic data: subtotals and item counts
        rng = np.random.default_rng(42)
        X = []
        y = []
        for _ in range(500):
            items = rng.integers(1, 15)
            subtotal = float(rng.uniform(5, 500))
            # Heuristic: larger orders more likely to want a discount
            prob = min(0.1 + 0.0015 * subtotal + 0.02 * items, 0.9)
            label = 1 if rng.random() < prob else 0
            X.append([subtotal, items])
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

    def predict_discount_prob(self, subtotal: float, num_items: int) -> float:
        if not SKLEARN_AVAILABLE or self.model is None:
            # Fallback heuristic if sklearn not available
            return min(0.1 + 0.0015 * subtotal + 0.02 * num_items, 0.9)
        X = np.array([[subtotal, num_items]])
        return float(self.model.predict_proba(X)[0, 1])


def build_invoice_from_user(items: List[Item], tax_rate: float, use_ai: bool) -> Invoice:
    inv = Invoice(items=items, tax_rate=tax_rate)
    if use_ai:
        model = SimpleDiscountModel()
        model.fit_demo()
        prob = model.predict_discount_prob(inv.subtotal, len(inv.items))
        if prob > 0.5:
            inv.discount_rate = DEFAULT_DISCOUNT_RATE
        print(f"[AI] Discount probability: {prob:.2f}. Applied: {inv.discount_rate>0}")
    return inv


def run_cli(args):
    print("AI Billing System (CLI)")
    print("Enter items. Leave name empty to finish.")
    items: List[Item] = []
    while True:
        name = input("Item name (blank to finish): ").strip()
        if not name:
            break
        try:
            price = float(input("Price: "))
            qty = int(input("Quantity: "))
        except ValueError:
            print("Invalid input. Please enter numeric price and integer quantity.")
            continue
        items.append(Item(name=name, price=price, quantity=qty))

    if not items:
        print("No items entered. Exiting.")
        return

    tax_rate = args.tax
    inv = build_invoice_from_user(items, tax_rate, args.ai)

    print("Summary:")
    for it in inv.items:
        print(f"- {it.name}: {it.quantity} x {it.price:.2f} = {it.price*it.quantity:.2f}")
    print(f"Subtotal: {inv.subtotal:.2f}")
    print(f"Tax @ {inv.tax_rate*100:.1f}%: {inv.tax:.2f}")
    if inv.discount_rate > 0:
        print(f"Discount @ {inv.discount_rate*100:.1f}%: -{inv.discount_amount:.2f}")
    print(f"TOTAL: {inv.total:.2f}")

    buyer = input("Buyer name (optional): ").strip() or "Customer"
    path = save_invoice_csv_txt(inv, buyer)
    print(f"Invoice saved: {path}")


class BillingGUI:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        root.title("AI Billing System")

        frm = ttk.Frame(root, padding=10)
        frm.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Table
        self.tree = ttk.Treeview(frm, columns=("price", "qty", "total"), show="headings", height=8)
        self.tree.heading("price", text="Price")
        self.tree.heading("qty", text="Qty")
        self.tree.heading("total", text="Line Total")
        self.tree.grid(row=0, column=0, columnspan=5, sticky="nsew")
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        # Inputs
        ttk.Label(frm, text="Item").grid(row=1, column=0, sticky="e")
        self.name_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.name_var, width=20).grid(row=1, column=1, sticky="w")

        ttk.Label(frm, text="Price").grid(row=1, column=2, sticky="e")
        self.price_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.price_var, width=10).grid(row=1, column=3, sticky="w")

        ttk.Label(frm, text="Qty").grid(row=1, column=4, sticky="e")
        self.qty_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.qty_var, width=5).grid(row=1, column=5, sticky="w")

        ttk.Button(frm, text="Add Item", command=self.add_item).grid(row=1, column=6, padx=5)

        # Summary labels
        self.subtotal_var = tk.StringVar(value="0.00")
        self.tax_var = tk.StringVar(value="0.00")
        self.discount_var = tk.StringVar(value="0.00")
        self.total_var = tk.StringVar(value="0.00")

        row2 = 2
        ttk.Label(frm, text="Subtotal:").grid(row=row2, column=5, sticky="e")
        ttk.Label(frm, textvariable=self.subtotal_var).grid(row=row2, column=6, sticky="w")
        row2 += 1
        ttk.Label(frm, text=f"Tax @ {self.args.tax*100:.1f}%:").grid(row=row2, column=5, sticky="e")
        ttk.Label(frm, textvariable=self.tax_var).grid(row=row2, column=6, sticky="w")
        row2 += 1
        ttk.Label(frm, text="Discount:").grid(row=row2, column=5, sticky="e")
        ttk.Label(frm, textvariable=self.discount_var).grid(row=row2, column=6, sticky="w")
        row2 += 1
        ttk.Label(frm, text="TOTAL:").grid(row=row2, column=5, sticky="e")
        ttk.Label(frm, textvariable=self.total_var, font=("TkDefaultFont", 10, "bold")).grid(row=row2, column=6, sticky="w")

        # Buyer + Actions
        ttk.Label(frm, text="Buyer:").grid(row=6, column=0, sticky="e", pady=(10,0))
        self.buyer_var = tk.StringVar(value="Customer")
        ttk.Entry(frm, textvariable=self.buyer_var, width=20).grid(row=6, column=1, sticky="w", pady=(10,0))
        ttk.Button(frm, text="Save Invoice", command=self.save).grid(row=6, column=6, sticky="e", pady=(10,0))

        self.items: List[Item] = []

    def add_item(self):
        try:
            name = self.name_var.get().strip()
            price = float(self.price_var.get())
            qty = int(self.qty_var.get())
            if not name or price < 0 or qty <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Input Error", "Please provide valid item, price, and quantity.")
            return
        self.items.append(Item(name, price, qty))
        self.tree.insert("", tk.END, values=(f"{price:.2f}", qty, f"{price*qty:.2f}"))
        self.name_var.set("")
        self.price_var.set("")
        self.qty_var.set("")
        self.refresh_totals()

    def refresh_totals(self):
        inv = build_invoice_from_user(self.items, self.args.tax, self.args.ai)
        self.subtotal_var.set(f"{inv.subtotal:.2f}")
        self.tax_var.set(f"{inv.tax:.2f}")
        self.discount_var.set(f"-{inv.discount_amount:.2f}")
        self.total_var.set(f"{inv.total:.2f}")
        self._last_invoice = inv

    def save(self):
        if not getattr(self, "_last_invoice", None):
            self.refresh_totals()
        buyer = self.buyer_var.get().strip() or "Customer"
        path = save_invoice_csv_txt(self._last_invoice, buyer)
        messagebox.showinfo("Saved", f"Invoice saved to: {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AI-Powered Billing System")
    p.add_argument("--tax", type=float, default=DEFAULT_TAX_RATE, help="Tax rate (e.g., 0.1 for 10%)")
    p.add_argument("--ai", action="store_true", help="Enable AI discount prediction")
    p.add_argument("--gui", action="store_true", help="Launch simple Tkinter GUI")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.gui:
        if not TK_AVAILABLE:
            print("Tkinter not available in this environment. Falling back to CLI.")
            run_cli(args)
        else:
            root = tk.Tk()
            BillingGUI(root, args)
            root.mainloop()
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
