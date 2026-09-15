import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================================
# تنظیمات پیش‌فرض
# ==========================================================
DEFAULT_LOG_PATH = "training_log_phase5_clean.json"  # یا training_log.jsonl
REFRESH_INTERVAL_MS = 1500             # نرخ به‌روزرسانی (میلی‌ثانیه)
WINDOW_SIZE = 10                       # اندازه پنجره برای میانگین متحرک (Smoothing)


def moving_average(data, window_size=10):
    """محاسبه میانگین متحرک جهت هموارسازی نوسانات RL."""
    if len(data) < window_size:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="valid")


def load_training_data(file_path):
    """
    خواندن مقاوم فایل JSON یا JSON-Lines حتی در صورت نوشتن هم‌زمان توسط پروسه دیگر.
    """
    if not os.path.exists(file_path):
        return []

    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []

            # تلاش برای خواندن به صورت JSON Array معمولی
            if content.startswith("[") and content.endswith("]"):
                try:
                    data = json.loads(content)
                    return data
                except json.JSONDecodeError:
                    pass

            # حالت JSON-Lines (هر خط یک دیکشنری مجزا)
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # چشم‌پوشی از خطوط ناقص ناشی از نوشتن زنده
    except Exception:
        return []

    return data


def update_plots(frame, file_path, axes):
    """تابع بازخوانی و به‌روزرسانی فریم‌های مت‌پلات‌لیب."""
    records = load_training_data(file_path)
    if not records:
        return

    # استخراج معیارها با فالبک مقادیر پیش‌فرض
    episodes = [r.get("episode", idx + 1) for idx, r in enumerate(records)]
    rewards = [r.get("reward", r.get("episode_reward", 0.0)) for r in records]
    ctes = [r.get("cte", r.get("mean_cte", 0.0)) for r in records]
    collisions = [1 if r.get("collision", False) else 0 for r in records]
    speeds = [r.get("speed", r.get("mean_speed", 0.0)) for r in records]

    for ax in axes.flat:
        ax.cla()
        ax.grid(True, linestyle="--", alpha=0.5)

    # 1. نمودار پاداش (Reward & Moving Average)
    ax_rew = axes[0, 0]
    ax_rew.plot(episodes, rewards, color="gray", alpha=0.35, label="Raw Reward")
    if len(rewards) >= WINDOW_SIZE:
        smooth_rewards = moving_average(rewards, WINDOW_SIZE)
        smooth_episodes = episodes[WINDOW_SIZE - 1:]
        ax_rew.plot(smooth_episodes, smooth_rewards, color="tab:blue", linewidth=2.0, label=f"MA ({WINDOW_SIZE})")
    ax_rew.set_title("Episode Reward")
    ax_rew.set_xlabel("Episode")
    ax_rew.set_ylabel("Reward")
    ax_rew.legend(loc="upper left")

    # 2. خطای جانبی مسیر (Cross-Track Error - CTE)
    ax_cte = axes[0, 1]
    ax_cte.plot(episodes, ctes, color="tab:orange", linewidth=1.5)
    if len(ctes) >= WINDOW_SIZE:
        smooth_cte = moving_average(ctes, WINDOW_SIZE)
        ax_cte.plot(episodes[WINDOW_SIZE - 1:], smooth_cte, color="darkred", linewidth=2.0, label=f"MA ({WINDOW_SIZE})")
    ax_cte.set_title("Cross-Track Error (CTE)")
    ax_cte.set_xlabel("Episode")
    ax_cte.set_ylabel("Meters")
    ax_cte.legend(loc="upper right")

    # 3. نرخ تجمعی برخورد (Collision Rate %)
    ax_col = axes[1, 0]
    if len(collisions) > 0:
        cum_collision_rate = (np.cumsum(collisions) / (np.arange(len(collisions)) + 1)) * 100.0
        ax_col.plot(episodes, cum_collision_rate, color="tab:red", linewidth=2.0)
        ax_col.set_ylim(0, 100)
    ax_col.set_title("Cumulative Collision Rate (%)")
    ax_col.set_xlabel("Episode")
    ax_col.set_ylabel("Collision %")

    # 4. میانگین سرعت خودرو (Vehicle Speed)
    ax_spd = axes[1, 1]
    ax_spd.plot(episodes, speeds, color="tab:green", linewidth=1.5)
    if len(speeds) >= WINDOW_SIZE:
        smooth_spd = moving_average(speeds, WINDOW_SIZE)
        ax_spd.plot(episodes[WINDOW_SIZE - 1:], smooth_spd, color="darkgreen", linewidth=2.0, label=f"MA ({WINDOW_SIZE})")
    ax_spd.set_title("Mean Vehicle Speed")
    ax_spd.set_xlabel("Episode")
    ax_spd.set_ylabel("km/h")
    ax_spd.legend(loc="upper left")

    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Live Training Metrics Monitor")
    parser.add_argument("--file", type=str, default=DEFAULT_LOG_PATH, help="Path to JSON/JSONL metrics file")
    parser.add_argument("--interval", type=int, default=REFRESH_INTERVAL_MS, help="Refresh rate in milliseconds")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.canvas.manager.set_window_title(f"Training Monitor — {args.file}")

    ani = FuncAnimation(
        fig,
        update_plots,
        fargs=(args.file, axes),
        interval=args.interval,
        cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
