from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemoDataSpec:
    seed: int = 7
    days: int = 60
    campaigns: int = 5
    adsets_per_campaign: int = 4
    ads_per_adset: int = 4
    leads_per_day: int = 200
    label_window_days: int = 14
    include_enrichments: bool = True


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def _stable_seed(s: str) -> int:
    # Deterministic across runs (unlike Python's built-in hash()).
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def generate_demo_data(output_dir: Path, spec: DemoDataSpec | None = None) -> dict[str, Path]:
    """
    Generate synthetic ads/leads/outcomes CSVs that match the canonical schemas.

    Returns a dict of {name: path}.
    """
    if spec is None:
        spec = DemoDataSpec()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)

    # --- Entity hierarchy ---
    campaigns = []
    adsets = []
    ads = []

    for ci in range(spec.campaigns):
        campaign_id = f"cmp_{ci:03d}"
        campaign_name = f"Campaign {ci:02d}"
        campaign_quality = rng.normal(loc=0.0, scale=1.0)
        campaigns.append((campaign_id, campaign_name, campaign_quality))
        for asi in range(spec.adsets_per_campaign):
            adset_id = f"as_{ci:03d}_{asi:03d}"
            adset_name = f"Adset {ci:02d}-{asi:02d}"
            adset_quality = campaign_quality + rng.normal(loc=0.0, scale=0.7)
            adsets.append((adset_id, adset_name, campaign_id, campaign_name, adset_quality))
            for ai in range(spec.ads_per_adset):
                ad_id = f"ad_{ci:03d}_{asi:03d}_{ai:03d}"
                ad_name = f"Ad {ci:02d}-{asi:02d}-{ai:02d}"
                ad_quality = adset_quality + rng.normal(loc=0.0, scale=0.5)
                ads.append(
                    (
                        ad_id,
                        ad_name,
                        adset_id,
                        adset_name,
                        campaign_id,
                        campaign_name,
                        ad_quality,
                    )
                )

    ads_df = pd.DataFrame(
        ads,
        columns=[
            "ad_id",
            "ad_name",
            "adset_id",
            "adset_name",
            "campaign_id",
            "campaign_name",
            "ad_quality",
        ],
    )

    # --- Date range ---
    # End "today" at UTC midnight to make label-maturity behavior deterministic.
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=spec.days - 1)
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D").date

    # --- Generate daily ads performance ---
    rows = []
    for d in all_dates:
        # Day-level seasonality (weekday effect)
        weekday = datetime(d.year, d.month, d.day, tzinfo=timezone.utc).weekday()
        day_mult = 1.0 + 0.15 * np.cos((weekday / 7.0) * 2 * np.pi)

        for _, r in ads_df.iterrows():
            q = float(r["ad_quality"])
            # Impressions driven by quality + noise
            imp_lambda = day_mult * (800 + 250 * max(q, -1.5) + rng.uniform(-80, 80))
            impressions = int(max(0, rng.poisson(lam=max(1.0, imp_lambda))))

            # CTR increases with quality a bit
            ctr = float(np.clip(0.008 + 0.004 * q + rng.normal(0, 0.002), 0.001, 0.05))
            clicks = int(rng.binomial(n=impressions, p=ctr)) if impressions > 0 else 0

            # CPC slightly lower for higher quality
            cpc = float(np.clip(2.2 - 0.4 * q + rng.normal(0, 0.2), 0.5, 6.0))
            spend = float(clicks * cpc)

            rows.append(
                {
                    "date": d.isoformat(),
                    "campaign_id": r["campaign_id"],
                    "campaign_name": r["campaign_name"],
                    "adset_id": r["adset_id"],
                    "adset_name": r["adset_name"],
                    "ad_id": r["ad_id"],
                    "ad_name": r["ad_name"],
                    "impressions": impressions,
                    "clicks": clicks,
                    "spend": round(spend, 2),
                }
            )

    ads_perf = pd.DataFrame(rows)

    # --- Generate leads ---
    # Allocate leads to ads proportionally to recent clicks.
    ads_perf["date_dt"] = pd.to_datetime(ads_perf["date"])
    latest_week = ads_perf[ads_perf["date_dt"] >= (ads_perf["date_dt"].max() - pd.Timedelta(days=6))]
    click_weights = (
        latest_week.groupby("ad_id")["clicks"].sum().reindex(ads_df["ad_id"]).fillna(0).to_numpy()
    )
    if click_weights.sum() <= 0:
        click_weights = np.ones_like(click_weights)
    probs = click_weights / click_weights.sum()

    lead_rows = []
    lead_id_seq = 0
    for d in all_dates:
        for _ in range(spec.leads_per_day):
            ad_idx = int(rng.choice(len(ads_df), p=probs))
            ad_row = ads_df.iloc[ad_idx]

            # Random time during the day (UTC)
            seconds = int(rng.integers(0, 24 * 3600))
            created_time = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(
                seconds=seconds
            )
            lead_id = f"lead_{lead_id_seq:09d}"
            lead_id_seq += 1

            lead_rows.append(
                {
                    "lead_id": lead_id,
                    "created_time": created_time.isoformat(),
                    "campaign_id": ad_row["campaign_id"],
                    "campaign_name": ad_row["campaign_name"],
                    "adset_id": ad_row["adset_id"],
                    "adset_name": ad_row["adset_name"],
                    "ad_id": ad_row["ad_id"],
                    "ad_name": ad_row["ad_name"],
                }
            )

    leads = pd.DataFrame(lead_rows)

    # --- Generate outcomes (qualified_time) ---
    # Qualification depends on ad quality + hour-of-day effect.
    leads_dt = pd.to_datetime(leads["created_time"], utc=True)
    hour = leads_dt.dt.hour.to_numpy()
    # hour effect: business hours slightly better
    hour_boost = np.where((hour >= 9) & (hour <= 17), 0.25, -0.05)

    lead_ads = leads.merge(ads_df[["ad_id", "ad_quality"]], on="ad_id", how="left")
    q = lead_ads["ad_quality"].fillna(0.0).to_numpy()
    logits = -1.8 + 0.9 * q + hour_boost + rng.normal(0, 0.6, size=len(leads))
    p = _sigmoid(logits)
    qualified = rng.random(len(leads)) < p

    qualified_time = []
    for is_q, ct in zip(qualified, leads_dt.to_list(), strict=True):
        if not is_q:
            qualified_time.append("")
            continue
        delta_days = int(rng.integers(0, spec.label_window_days + 1))
        delta_hours = int(rng.integers(0, 24))
        qt = ct + pd.Timedelta(days=delta_days, hours=delta_hours)
        # keep within the label window
        if qt > ct + pd.Timedelta(days=spec.label_window_days):
            qt = ct + pd.Timedelta(days=spec.label_window_days)
        qualified_time.append(qt.isoformat())

    outcomes = pd.DataFrame({"lead_id": leads["lead_id"], "qualified_time": qualified_time})

    # --- Optional enrichment signals ---
    ads_placement = None
    ads_geo = None
    adset_targeting = None
    ad_creatives = None

    if spec.include_enrichments:
        # Creative type per ad (static)
        creative_types = []
        for _, r in ads_df.iterrows():
            q = float(r["ad_quality"])
            # Higher quality slightly more likely to be video.
            p_video = float(np.clip(_sigmoid(np.array([q]))[0], 0.15, 0.85))
            ct = "video" if rng.random() < p_video else "image"
            if rng.random() < 0.05:
                ct = "carousel"
            creative_types.append({"ad_id": r["ad_id"], "creative_type": ct})
        ad_creatives = pd.DataFrame(creative_types)

        # Audience keywords per adset (static)
        keyword_pool = [
            "small business",
            "entrepreneurship",
            "marketing",
            "digital marketing",
            "lead generation",
            "facebook ads",
            "instagram",
            "ecommerce",
            "shopify",
            "real estate",
            "mortgage",
            "home improvement",
            "fitness",
            "personal training",
            "weight loss",
            "nutrition",
            "dentist",
            "chiropractor",
            "med spa",
            "cosmetic surgery",
            "wedding",
            "travel",
            "insurance",
            "accounting",
            "law firm",
            "solar",
            "roofing",
            "plumbing",
            "hair salon",
            "yoga",
            "saas",
            "crm",
            "online course",
            "coaching",
        ]
        adset_rows = []
        for adset_id, adset_name, campaign_id, campaign_name, adset_quality in adsets:
            # Theme per campaign for a bit more structure
            rng2 = np.random.default_rng(_stable_seed(str(campaign_id)))
            theme = rng2.choice(keyword_pool, size=8, replace=False).tolist()
            k = int(rng.integers(4, 10))
            kws = rng.choice(theme, size=min(k, len(theme)), replace=False).tolist()
            adset_rows.append({"adset_id": adset_id, "audience_keywords": ", ".join(kws)})
        adset_targeting = pd.DataFrame(adset_rows)

        # Daily placement/geo breakdowns derived from ads_perf (so sums match)
        placements = ["feed", "stories", "reels"]
        geos = ["US-CA", "US-TX", "US-NY", "US-FL", "US-IL"]

        # Base geo mixture per campaign
        camp_geo = {}
        for cid, _, _q in campaigns:
            camp_geo[cid] = rng.dirichlet(np.array([2.0] * len(geos)))

        placement_rows = []
        geo_rows = []
        for _, r in ads_perf.iterrows():
            impressions = int(r["impressions"])
            clicks = int(r["clicks"])
            spend = float(r["spend"])

            # Placements: mild per-row noise
            alpha = np.array([2.0, 1.6, 1.3])
            p_place = rng.dirichlet(alpha)
            imp_split = rng.multinomial(impressions, p_place) if impressions > 0 else np.zeros(len(placements), dtype=int)
            clk_split = rng.multinomial(clicks, p_place) if clicks > 0 else np.zeros(len(placements), dtype=int)
            sp_split = (p_place * spend).astype(float)
            sp_split = np.round(sp_split, 2)
            if len(sp_split):
                sp_split[-1] = round(float(spend) - float(sp_split[:-1].sum()), 2)

            for i, pl in enumerate(placements):
                placement_rows.append(
                    {
                        "date": r["date"],
                        "ad_id": r["ad_id"],
                        "placement": pl,
                        "impressions": int(imp_split[i]),
                        "clicks": int(clk_split[i]),
                        "spend": float(sp_split[i]),
                    }
                )

            # Geos: mostly campaign-level mixture, with a little per-row noise
            base = camp_geo.get(str(r["campaign_id"]), np.array([1.0 / len(geos)] * len(geos)))
            noise = rng.dirichlet(np.array([1.5] * len(geos)))
            p_geo = (0.75 * base) + (0.25 * noise)
            p_geo = p_geo / p_geo.sum()
            imp_split = rng.multinomial(impressions, p_geo) if impressions > 0 else np.zeros(len(geos), dtype=int)
            clk_split = rng.multinomial(clicks, p_geo) if clicks > 0 else np.zeros(len(geos), dtype=int)
            sp_split = (p_geo * spend).astype(float)
            sp_split = np.round(sp_split, 2)
            if len(sp_split):
                sp_split[-1] = round(float(spend) - float(sp_split[:-1].sum()), 2)

            for i, g in enumerate(geos):
                geo_rows.append(
                    {
                        "date": r["date"],
                        "ad_id": r["ad_id"],
                        "geo": g,
                        "impressions": int(imp_split[i]),
                        "clicks": int(clk_split[i]),
                        "spend": float(sp_split[i]),
                    }
                )

        ads_placement = pd.DataFrame(placement_rows)
        ads_geo = pd.DataFrame(geo_rows)

    # --- Write CSVs ---
    ads_path = output_dir / "ads.csv"
    leads_path = output_dir / "leads.csv"
    outcomes_path = output_dir / "outcomes.csv"
    ads_perf.drop(columns=["date_dt"]).to_csv(ads_path, index=False)
    leads.to_csv(leads_path, index=False)
    outcomes.to_csv(outcomes_path, index=False)

    out = {"ads": ads_path, "leads": leads_path, "outcomes": outcomes_path}
    if spec.include_enrichments:
        if ads_placement is not None:
            p = output_dir / "ads_placement.csv"
            ads_placement.to_csv(p, index=False)
            out["ads_placement"] = p
        if ads_geo is not None:
            p = output_dir / "ads_geo.csv"
            ads_geo.to_csv(p, index=False)
            out["ads_geo"] = p
        if adset_targeting is not None:
            p = output_dir / "adset_targeting.csv"
            adset_targeting.to_csv(p, index=False)
            out["adset_targeting"] = p
        if ad_creatives is not None:
            p = output_dir / "ad_creatives.csv"
            ad_creatives.to_csv(p, index=False)
            out["ad_creatives"] = p

    return out
