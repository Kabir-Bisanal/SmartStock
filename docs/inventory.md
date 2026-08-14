# Inventory decision methodology

## Data disclosure

Walmart M5 provides unit sales, calendar context, and weekly prices—not actual on-hand inventory, purchase orders, supplier lead times, service targets, or costs. SmartStock therefore uses deterministic synthetic demo values for those fields. They are useful for exercising the decision engine but are not Walmart operational data and do not prove real savings.

## Core quantities

- **Inventory position** = on-hand + on-order − backorders.
- **Lead-time demand** sums daily forecasts across the selected lead time.
- **Forecast uncertainty** is the standard deviation of `actual − forecast` residuals from validation-only 28-day-mean predictions, with documented fallback levels.
- **Safety stock** = service-level normal quantile × daily residual standard deviation × square root of lead-time days.
- **Reorder point** = lead-time forecast demand + safety stock.
- **Target stock** covers lead time plus a seven-day review period, plus safety stock.
- **Service-level order** = max(0, target stock − inventory position), rounded up to whole units.

Stockout risk uses a normal approximation for demand during lead time. Status precedence distinguishes STOCKOUT, CRITICAL, REORDER NOW, LOW, OVERSTOCK, and HEALTHY. Priority combines stockout risk, reorder-point gap, and target shortage into a 0–100 score.

## Cost-aware scenario

The optional cost recommendation evaluates a bounded set of order quantities against simulated demand scenarios. Its illustrative objective combines fixed order cost, shortage cost, and approximate holding cost. Random seed 42 makes the demonstration reproducible. It does not include purchase cost, capacity, supplier minimums, case packs, shelf life, or correlated demand error.

## V1 output

Across 300 item-store pairs, the saved demo produced 198 service-level reorders totaling 4,143 units. A separate cost view selected 136 orders totaling 5,181 units. Expected shortage fell from 3,312.6 to 167.0 demo units. Illustrative expected cost fell from 16,937.5 to 4,892.6 demo currency units. The implied 12,044.8 is simulated analytical output—not realized business savings.

The reusable calculation entry point is `smartstock.inventory.engine.recommend_inventory`; the UI does not reimplement its formulas.
