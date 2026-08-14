import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from ftmoquant.research.tournament_dashboard import build_dashboard_snapshot

    return build_dashboard_snapshot, mo


@app.cell
def _(build_dashboard_snapshot):
    # This view model has fixed DEVELOPMENT roots and exposes no return reader,
    # validation accessor, holdout accessor, or arbitrary path control.
    snapshot = build_dashboard_snapshot()
    return (snapshot,)


@app.cell
def _(mo):
    mo.md("""
    # G1.4B development tournament infrastructure

    Infrastructure status only. Candidate performance, strategy returns, equity
    curves, validation data, and final-holdout data are unavailable in this app.
    """)
    return


@app.cell
def _(mo, snapshot):
    mo.md(
        f"""
        ## Frozen universe

        - Universe: `{snapshot["universe_id"]}`
        - Readiness: **{snapshot["frozen_readiness"]}**
        - Readiness SHA: `{snapshot["universe_readiness_sha256"]}`
        - Plan SHA: `{snapshot["universe_plan_sha256"]}`
        - Ordered instruments: `{", ".join(snapshot["ordered_instruments"])}`
        - DEVELOPMENT: `{snapshot["development_interval"][0]}` to
          `{snapshot["development_interval"][1]}` exclusive
        """
    )
    return


@app.cell
def _(mo, snapshot):
    mo.vstack(
        [
            mo.md("## Currency incidence and exposure metadata"),
            mo.ui.table(list(snapshot["currency_metadata"]), selection=None),
        ]
    )
    return


@app.cell
def _(mo, snapshot):
    mo.vstack(
        [
            mo.md(
                f"""
        ## DEVELOPMENT availability and synchronization

        **{snapshot["synchronization"]}**

        Only split manifests and catalog-directory availability are shown. This page
        does not scan market rows or calculate candidate returns.
                """
            ),
            mo.ui.table(list(snapshot["development_artifacts"]), selection=None),
        ]
    )
    return


@app.cell
def _(mo, snapshot):
    mo.vstack(
        [
            mo.md("## Tournament registry eligibility"),
            mo.ui.table(list(snapshot["registry"]), selection=None),
        ]
    )
    return


@app.cell
def _(mo, snapshot):
    mo.md(
        f"""
        ## Locked boundaries

        - Validation: **{snapshot["validation"]}**
        - Final holdout: **{snapshot["holdout"]}**
        - Strategy returns: **{snapshot["strategy_returns"]}**
        - Fold contract SHA: `{snapshot["folds_sha256"]}`
        - Registry SHA: `{snapshot["registry_sha256"]}`
        - Selection contract SHA: `{snapshot["selection_contract_sha256"]}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
