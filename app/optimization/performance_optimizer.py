import time
from typing import Any, Dict, Optional


class PerformanceOptimizer:
    """
    Stage 23:
    Performance monitoring and optimization support.

    This component does not change the QA logic.
    It measures execution time of pipeline stages and
    provides a compact performance report.

    Safe design:
    - Does not modify claims.
    - Does not modify evidence.
    - Does not modify answers.
    - Does not modify citations.
    - Does not modify confidence scores.
    """

    def __init__(self):
        self.stage_times: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    # ============================================================
    # START TIMER
    # ============================================================

    def start(self, stage: str) -> None:
        """
        Start timing a pipeline stage.
        """
        stage = str(stage).strip()

        if not stage:
            return

        self._start_times[stage] = time.perf_counter()

    # ============================================================
    # STOP TIMER
    # ============================================================

    def stop(self, stage: str) -> float:
        """
        Stop timing a pipeline stage.

        Returns elapsed time in seconds.
        """
        stage = str(stage).strip()

        if not stage:
            return 0.0

        start_time = self._start_times.get(stage)

        if start_time is None:
            return 0.0

        elapsed = time.perf_counter() - start_time

        self.stage_times[stage] = round(
            elapsed,
            4
        )

        self._start_times.pop(stage, None)

        return elapsed

    # ============================================================
    # MEASURE CONTEXT
    # ============================================================

    def measure(self, stage: str):
        """
        Context-manager helper for timing a stage.

        Example:

            with optimizer.measure("retrieval"):
                results = retriever.retrieve(...)
        """

        return _PerformanceTimer(
            self,
            stage
        )

    # ============================================================
    # RECORD EXTERNAL TIME
    # ============================================================

    def record(
        self,
        stage: str,
        seconds: float
    ) -> None:
        """
        Directly record an execution time.
        """
        stage = str(stage).strip()

        if not stage:
            return

        self.stage_times[stage] = round(
            float(seconds),
            4
        )

    # ============================================================
    # TOTAL TIME
    # ============================================================

    def total_time(self) -> float:
        """
        Return total measured pipeline time.
        """
        return round(
            sum(self.stage_times.values()),
            4
        )

    # ============================================================
    # SLOWEST STAGE
    # ============================================================

    def slowest_stage(self) -> Optional[str]:
        """
        Return the name of the slowest measured stage.
        """
        if not self.stage_times:
            return None

        return max(
            self.stage_times,
            key=self.stage_times.get
        )

    # ============================================================
    # PERFORMANCE REPORT
    # ============================================================

    def report(self) -> Dict[str, Any]:
        """
        Return a structured performance report.
        """

        total = self.total_time()

        stages = {}

        for stage, seconds in self.stage_times.items():

            percentage = 0.0

            if total > 0:
                percentage = (
                    seconds / total
                ) * 100.0

            stages[stage] = {
                "seconds": round(
                    seconds,
                    4
                ),
                "percentage": round(
                    percentage,
                    2
                ),
            }

        return {
            "total_seconds": total,
            "stage_count": len(
                self.stage_times
            ),
            "slowest_stage": self.slowest_stage(),
            "stages": stages,
        }

    # ============================================================
    # PRINT REPORT
    # ============================================================

    def print_report(self) -> None:
        """
        Print a human-readable performance report.
        """

        report = self.report()

        print(
            "\n" + "=" * 80
        )

        print(
            "PERFORMANCE REPORT"
        )

        print(
            "=" * 80
        )

        print(
            "Total measured time:",
            report["total_seconds"],
            "seconds"
        )

        print(
            "Measured stages:",
            report["stage_count"]
        )

        print(
            "Slowest stage:",
            report["slowest_stage"]
        )

        print(
            "\nStage timings:"
        )

        for stage, data in report[
            "stages"
        ].items():

            print(
                f"  {stage}: "
                f"{data['seconds']} sec "
                f"({data['percentage']}%)"
            )

        print(
            "=" * 80
        )


# ================================================================
# INTERNAL TIMER
# ================================================================

class _PerformanceTimer:
    """
    Internal context manager used by PerformanceOptimizer.
    """

    def __init__(
        self,
        optimizer: PerformanceOptimizer,
        stage: str
    ):
        self.optimizer = optimizer
        self.stage = stage

    def __enter__(self):
        self.optimizer.start(
            self.stage
        )

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):
        self.optimizer.stop(
            self.stage
        )

        return False

