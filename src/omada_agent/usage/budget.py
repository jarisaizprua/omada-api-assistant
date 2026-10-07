"""Spending limits.

Four independent guards, checked *before* a question is sent to the model:

* ``BUDGET_DAILY_USD``       everybody together, per day
* ``BUDGET_MONTHLY_USD``     everybody together, per month
* ``BUDGET_USER_DAILY_USD``  each person, per day
* ``USER_DAILY_REQUESTS``    each person, questions per day (works even when the model's price is unknown)

A value of 0 switches that guard off. The check happens before the request, so one answer can
overshoot a limit by its own cost (cents); it can never run away.
"""

from __future__ import annotations

from dataclasses import dataclass

from omada_agent.config import Settings
from omada_agent.usage.store import UsageStore

WARN_AT = 0.8  # tell the user when they have used this share of a limit


@dataclass(slots=True)
class BudgetDecision:
    allowed: bool
    code: str = ""
    message: str = ""


def _usd(value: float) -> str:
    return f"US$ {value:,.2f}"


class BudgetGuard:
    def __init__(self, settings: Settings, store: UsageStore):
        self.settings = settings
        self.store = store

    # ------------------------------------------------------------------ snapshot
    def snapshot(self, user: str) -> dict:
        s, st = self.settings, self.store
        day, month = st.day_key(), st.month_key()
        today_all = st.spent(day=day)
        month_all = st.spent(month=month)
        today_user = st.spent(day=day, user=user)
        limits = {
            "daily_usd": s.budget_daily_usd,
            "monthly_usd": s.budget_monthly_usd,
            "user_daily_usd": s.budget_user_daily_usd,
            "user_daily_requests": s.user_daily_requests,
        }
        used = {
            "daily_usd": today_all["cost"],
            "monthly_usd": month_all["cost"],
            "user_daily_usd": today_user["cost"],
            "user_daily_requests": today_user["requests"],
        }
        shares = {k: (used[k] / limits[k]) if limits[k] > 0 else 0.0 for k in limits}
        return {
            "timezone": st.tz_name,
            "day": day,
            "month": month,
            "limits": limits,
            "used": used,
            "share": shares,
            "highest_share": max(shares.values()) if shares else 0.0,
        }

    # ------------------------------------------------------------------ decision
    def check(self, user: str) -> BudgetDecision:
        snap = self.snapshot(user)
        limits, used = snap["limits"], snap["used"]
        tz = snap["timezone"]

        if limits["user_daily_requests"] > 0 and used["user_daily_requests"] >= limits["user_daily_requests"]:
            return BudgetDecision(
                False,
                "requests_user_daily",
                f"Llegaste a tu límite de {limits['user_daily_requests']} preguntas por día. "
                f"Se restablece a medianoche (zona horaria {tz}).",
            )
        if limits["user_daily_usd"] > 0 and used["user_daily_usd"] >= limits["user_daily_usd"]:
            return BudgetDecision(
                False,
                "budget_user_daily",
                f"Llegaste a tu límite diario de uso ({_usd(limits['user_daily_usd'])}). "
                f"Se restablece a medianoche (zona horaria {tz}).",
            )
        if limits["daily_usd"] > 0 and used["daily_usd"] >= limits["daily_usd"]:
            return BudgetDecision(
                False,
                "budget_daily",
                f"El asistente alcanzó su presupuesto diario ({_usd(limits['daily_usd'])}). "
                f"Vuelve a estar disponible a medianoche (zona horaria {tz}) o pide al administrador subir BUDGET_DAILY_USD.",
            )
        if limits["monthly_usd"] > 0 and used["monthly_usd"] >= limits["monthly_usd"]:
            return BudgetDecision(
                False,
                "budget_monthly",
                f"El asistente alcanzó su presupuesto mensual ({_usd(limits['monthly_usd'])}). "
                "Se reactiva el mes próximo o cuando el administrador suba BUDGET_MONTHLY_USD.",
            )
        return BudgetDecision(True)

    def warning(self, snapshot: dict) -> str | None:
        """A friendly heads-up once any limit is at 80 % or more."""
        names = {
            "user_daily_requests": "tu límite diario de preguntas",
            "user_daily_usd": "tu límite diario de uso",
            "daily_usd": "el presupuesto diario del asistente",
            "monthly_usd": "el presupuesto mensual del asistente",
        }
        key, share = max(snapshot["share"].items(), key=lambda kv: kv[1])
        if share >= WARN_AT:
            return f"Ya se usó el {min(int(share * 100), 100)} % de {names[key]}."
        return None
