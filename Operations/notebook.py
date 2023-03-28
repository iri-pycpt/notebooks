import cartopy
import cartopy.crs as ccrs
import cptcore as cc
import cptdl as dl
import cptextras as ce
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.stats import norm, t
import xarray as xr


missing_value_flag = -999


def setup(download_args, caseDir):
    # extracting domain boundaries and create house keeping
    domain = download_args["predictor_extent"]
    e, w, n, s = domain.values()

    domainFolder = str(w) + "W-" + str(e) + "E" + "_to_" + str(s) + "S-" + str(n) + "N"

    files_root = Path.home() / "Desktop" / caseDir / domainFolder
    files_root.mkdir(exist_ok=True, parents=True)

    dataDir = files_root / "data"
    dataDir.mkdir(exist_ok=True, parents=True)

    figDir = files_root / "figures"
    figDir.mkdir(exist_ok=True, parents=True)

    outputDir = files_root / "output"
    outputDir.mkdir(exist_ok=True, parents=True)

    print(f"Input data will be saved in {dataDir}")
    print(f"Figures will be saved in {figDir}")
    print(f"Output will be saved in {outputDir}")

    return files_root


def download_data(
        predictand_name, predictor_names, download_args, files_root, force_download
):
    Y, graph_orientation = download_observations(
        download_args, files_root, predictand_name, force_download
    )
    hindcast_data = download_hindcasts(
        predictor_names, files_root, force_download, download_args, Y
    )
    forecast_data = download_forecasts(
        predictor_names, files_root, force_download, download_args, Y
    )
    return Y, hindcast_data, forecast_data, graph_orientation


def download_observations(download_args, files_root, predictand_name, force_download):
    dataDir = files_root / "data"
    # Deal with "Cross-year issues" where either the target season
    # crosses Jan 1 (eg DJF), or where the forecast initialization is
    # in the calendar year before the start of the target season (eg
    # JFM from Dec 1 sart)

    fmon = download_args["fdate"].month
    tmon1 = fmon + download_args["lead_low"]  # first month of the target season
    tmon2 = fmon + download_args["lead_high"]  # last month of the target season
    download_args_obs = download_args.copy()

    # For when the target season crossing Jan 1 (eg DJF)
    # (i.e., when target season starts in the same calendar year as the forecast init
    # and ends in the following calendar year)
    # Here the final year of the obs dataset needs to be incremented by 1.
    if tmon1 <= 12.5 and tmon2 > 12.5:
        download_args_obs["final_year"] += 1

    # For JFM, FMA .. with forecast initialization in the previous year.
    # (i.e., when target season starts in the calendar year after the forecast init.)
    # Here both the first and final year of the obs dataset need to be incremented by 1.
    if tmon1 > 12.5:
        download_args_obs["first_year"] += 1
        download_args_obs["final_year"] += 1

    print(download_args)
    print(download_args_obs)

    if not Path(dataDir / "{}.nc".format(predictand_name)).is_file() or force_download:
        Y = dl.download(
            dl.observations[predictand_name],
            dataDir / (predictand_name + ".tsv"),
            **download_args_obs,
            verbose=True,
            use_dlauth=False,
        )
        Y = getattr(Y, [i for i in Y.data_vars][0])
        Y.to_netcdf(dataDir / "{}.nc".format(predictand_name))
    else:
        Y = xr.open_dataset(dataDir / "{}.nc".format(predictand_name))
        Y = getattr(Y, [i for i in Y.data_vars][0])

    graph_orientation = ce.graphorientation(len(Y["X"]), len(Y["Y"]))
    return Y, graph_orientation


def download_hindcasts(predictor_names, files_root, force_download, download_args, Y):
    dataDir = files_root / "data"
    # download training data
    hindcast_data = []
    for model in predictor_names:
        if not Path(dataDir / (model + ".nc")).is_file() or force_download:
            X = dl.download(
                dl.hindcasts[model],
                dataDir / (model + ".tsv"),
                **download_args,
                verbose=True,
                use_dlauth=False,
            )
            X = getattr(X, [i for i in X.data_vars][0])
            X.name = Y.name
            X.to_netcdf(dataDir / "{}.nc".format(model))
        else:
            X = xr.open_dataset(dataDir / (model + ".nc"))
            X = getattr(X, [i for i in X.data_vars][0])
            X.name = Y.name
        hindcast_data.append(X)
    return hindcast_data


def download_forecasts(predictor_names, files_root, force_download, download_args, Y):
    dataDir = files_root / "data"
    forecast_data = []
    for model in predictor_names:
        if not Path(dataDir / (model + "_f.nc")).is_file() or force_download:
            F = dl.download(
                dl.forecasts[model],
                dataDir / (model + "_f.tsv"),
                **download_args,
                verbose=True,
                use_dlauth=False,
            )
            F = getattr(F, [i for i in F.data_vars][0])
            F.name = Y.name
            F.to_netcdf(dataDir / (model + "_f.nc"))
        else:
            F = xr.open_dataset(dataDir / (model + "_f.nc"))
            F = getattr(F, [i for i in F.data_vars][0])
            F.name = Y.name
        forecast_data.append(F)
    return forecast_data


def plot_skill(predictor_names, skill, MOS, files_root):
    # determnistic skill metrics: 'pearson', 'spearman', 'two_alternative_forced_choice', 'roc_area_under_curve', 'roc_area_above_curve'
    # roc_area_under_curve = ROC Below Normal category
    # roc_area_above_curve = ROC Above Normal category
    # probabilistic skill metrics (in sample): 'generalized_roc', 'ignorance', 'rank_probability_skill_score'

    # cmaps = [plt.get_cmap('cpt.correlation', 11), plt.get_cmap('cpt.correlation', 11), plt.get_cmap('RdBu_r', 11), plt.get_cmap('cpt.correlation', 11) ]
    # limits = [(-1, 1), (-1, 1), (0, 100), (-50, 50)]

    # deterministic five:
    skill_metrics = [
        "pearson",
        "spearman",
        "two_alternative_forced_choice",
        "roc_area_below_normal",
        "roc_area_above_normal",
    ]
    cmaps = [
        ce.cmaps["cpt_correlation"],
        ce.cmaps["cpt_correlation"],
        ce.cmaps["pycpt_roc"],
        ce.cmaps["pycpt_roc"],
        ce.cmaps["pycpt_roc"],
    ]
    limits = [(-1, 1), (-1, 1), (0, 100), (0, 1), (0, 1)]

    fig, ax = plt.subplots(
        nrows=len(predictor_names),
        ncols=len(skill_metrics),
        subplot_kw={"projection": cartopy.crs.PlateCarree()},
        figsize=(5 * len(skill_metrics), 2.5 * len(predictor_names)),
    )
    if len(predictor_names) == 1:
        ax = [ax]

    for i, model in enumerate(predictor_names):
        for j, skill_metric in enumerate(skill_metrics):
            n = (
                getattr(skill[i], skill_metric)
                .where(getattr(skill[i], skill_metric) > missing_value_flag)
                .plot(ax=ax[i][j], cmap=cmaps[j], vmin=limits[j][0], vmax=limits[j][1])
            )
            ax[i][j].coastlines()
            ax[i][j].add_feature(cartopy.feature.BORDERS)
            ax[0][j].set_title(skill_metric.upper())

        ax[i][0].text(
            -0.07,
            0.55,
            model.upper(),
            va="bottom",
            ha="center",
            rotation="vertical",
            rotation_mode="anchor",
            transform=ax[i][0].transAxes,
        )

    # save plots
    figName = MOS + "_models_skillMatrices.png"
    fig.savefig(
        files_root / "figures" / figName,
        bbox_inches="tight",
    )


def plot_cca_modes(
    MOS, predictor_names, pxs, pys, graph_orientation, files_root
):
    nmodes = 3
    cmap = plt.get_cmap("cpt.loadings", 11)
    vmin = -10
    vmax = 10
    missing_value_flag = -999

    if MOS == "CCA":
        for i, model in enumerate(predictor_names):
            for mode in range(nmodes):
                if mode == 0 and model == predictor_names[0]:
                    Vmin, Vmax = ce.standardized_range(
                        float(
                            pxs[i]
                            .x_cca_loadings.isel(Mode=mode)
                            .where(
                                pxs[i].x_cca_loadings.isel(Mode=mode)
                                > missing_value_flag
                            )
                            .min()
                        ),
                        float(
                            pxs[i]
                            .x_cca_loadings.isel(Mode=mode)
                            .where(
                                pxs[i].x_cca_loadings.isel(Mode=mode)
                                > missing_value_flag
                            )
                            .max()
                        ),
                    )

                cancorr = np.correlate(
                    pxs[i].x_cca_scores[:, mode], pys[i].y_cca_scores[:, mode]
                )
                print(
                    model.upper()
                    + ": CCA MODE {}".format(mode + 1)
                    + " - Canonical Correlation = "
                    + str(ce.truncate(cancorr[0], 2))
                )

                fig = plt.figure(figsize=(20, 5))

                gs0 = gridspec.GridSpec(1, 3, figure=fig)
                gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])
                gs01 = gridspec.GridSpecFromSubplotSpec(5, 10, subplot_spec=gs0[1])
                gs02 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[2])
                ts = xr.concat(
                    [
                        pxs[i].x_cca_scores.isel(Mode=mode),
                        pys[i].y_cca_scores.isel(Mode=mode),
                    ],
                    "M",
                ).assign_coords({"M": ["x", "y"]})

                map1_ax = fig.add_subplot(gs00[:, :], projection=ccrs.PlateCarree())
                ts_ax = fig.add_subplot(gs01[1:3, 1:])
                map2_ax = fig.add_subplot(gs02[:, :], projection=ccrs.PlateCarree())

                art = (
                    pxs[i]
                    .x_cca_loadings.isel(Mode=mode)
                    .where(pxs[i].x_cca_loadings.isel(Mode=mode) > missing_value_flag)
                    .plot(
                        ax=map1_ax, add_colorbar=False, vmin=Vmin, vmax=Vmax, cmap=cmap
                    )
                )

                cb = plt.colorbar(art, orientation=graph_orientation)
                cb.set_label(label="x_cca_loadings", size=14)
                cb.ax.tick_params(labelsize=12)

                art = (
                    pys[i]
                    .y_cca_loadings.isel(Mode=mode)
                    .where(pys[i].y_cca_loadings.isel(Mode=mode) > missing_value_flag)
                    .plot(
                        ax=map2_ax, add_colorbar=False, vmin=Vmin, vmax=Vmax, cmap=cmap
                    )
                )
                cb = plt.colorbar(art, orientation=graph_orientation)
                cb.set_label(label="y_cca_loadings", size=14)
                cb.ax.tick_params(labelsize=12)

                primitive = ts.plot.line(
                    marker="x", ax=ts_ax, markersize=12, hue="M", add_legend=False
                )
                ts_ax.grid(axis="x", linestyle="-.")
                ts_ax.legend(
                    handles=primitive, labels=list(ts.coords["M"].values), loc="best"
                )
                ts_ax.spines["top"].set_visible(False)
                ts_ax.spines["right"].set_visible(False)
                ts_ax.spines["bottom"].set_visible(False)
                ts_ax.set_title("CCA Scores (Mode {})".format(mode + 1))
                ts_ax.set_ylabel(None)
                ts_ax.set_xlabel(None)

                map1_ax.set_title("X CCA MODE {}".format(mode + 1))
                map2_ax.set_title("Y CCA MODE {}".format(mode + 1))

                map1_ax.coastlines()
                map2_ax.coastlines()

                map1_ax.add_feature(cartopy.feature.BORDERS)
                map2_ax.add_feature(cartopy.feature.BORDERS)
                plt.show()

                # save plots
                figName = MOS + "_" + str(model) + "_CCA_mode_" + str(mode + 1) + ".png"
                fig.savefig(files_root / "figures" / figName, bbox_inches="tight")
    else:
        print("You will need to set MOS=CCA in order to see CCA Modes")


def plot_eof_modes(
    MOS, predictor_names, pxs, pys, graph_orientation, files_root
):
    nmodes = 5
    cmap = plt.get_cmap("cpt.loadings", 11)
    vmin = -10
    vmax = 10

    import matplotlib.ticker as mticker
    import matplotlib.gridspec as gridspec

    if MOS == "CCA":
        for i, model in enumerate(predictor_names):
            for mode in range(nmodes):
                if mode == 0 and model == predictor_names[0]:
                    Vmin, Vmax = ce.standardized_range(
                        float(
                            pxs[i]
                            .x_eof_loadings.isel(Mode=mode)
                            .where(
                                pxs[i].x_eof_loadings.isel(Mode=mode)
                                > missing_value_flag
                            )
                            .min()
                        ),
                        float(
                            pxs[i]
                            .x_eof_loadings.isel(Mode=mode)
                            .where(
                                pxs[i].x_eof_loadings.isel(Mode=mode)
                                > missing_value_flag
                            )
                            .max()
                        ),
                    )

                print(
                    model.upper() + ": EOF {}".format(mode + 1)
                )  # str(truncate(canvar[0], 2)))
                fig = plt.figure(figsize=(20, 5))

                gs0 = gridspec.GridSpec(1, 3, figure=fig)
                gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])
                gs01 = gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=gs0[1])
                gs02 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[2])
                ts = xr.concat(
                    [
                        pxs[i].x_eof_scores.isel(Mode=mode),
                        pys[i].y_eof_scores.isel(Mode=mode),
                    ],
                    "M",
                ).assign_coords({"M": ["x", "y"]})

                map1_ax = fig.add_subplot(gs00[:, :], projection=ccrs.PlateCarree())
                ts_ax = fig.add_subplot(gs01[1:3, 1:])
                map2_ax = fig.add_subplot(gs02[:, :], projection=ccrs.PlateCarree())

                art = (
                    pxs[i]
                    .x_eof_loadings.isel(Mode=mode)
                    .where(pxs[i].x_eof_loadings.isel(Mode=mode) > missing_value_flag)
                    .plot(
                        ax=map1_ax, add_colorbar=False, vmin=Vmin, vmax=Vmax, cmap=cmap
                    )
                )

                canvarX = round(
                    float(
                        pxs[i]
                        .x_explained_variance.isel(Mode=mode)
                        .where(
                            pxs[i].x_explained_variance.isel(Mode=mode)
                            > missing_value_flag
                        )
                        .max()
                    ),
                    1,
                )

                cb = plt.colorbar(art, orientation=graph_orientation)
                cb.set_label(label="x_eof_loadings", size=14)
                cb.ax.tick_params(labelsize=12)
                if graph_orientation == "horizontal":
                    cb.ax.tick_params(axis="x", which="major", rotation=-45)
                # cb.ax.set_xticks(len(6))
                pxs[i].x_explained_variance
                art = (
                    pys[i]
                    .y_eof_loadings.isel(Mode=mode)
                    .where(pys[i].y_eof_loadings.isel(Mode=mode) > missing_value_flag)
                    .plot(
                        ax=map2_ax, add_colorbar=False, vmin=Vmin, vmax=Vmax, cmap=cmap
                    )
                )

                canvarY = round(
                    float(
                        pys[i]
                        .y_explained_variance.isel(Mode=mode)
                        .where(
                            pys[i].y_explained_variance.isel(Mode=mode)
                            > missing_value_flag
                        )
                        .max()
                    ),
                    1,
                )
                cb = plt.colorbar(art, orientation=graph_orientation)
                ticks_loc = cb.ax.get_xticklabels()
                cb.set_label(label="y_eof_loadings", size=14)
                cb.ax.tick_params(labelsize=12)

                if graph_orientation == "horizontal":
                    cb.ax.tick_params(axis="x", which="major", rotation=-45)

                primitive = ts.plot.line(
                    marker="x", ax=ts_ax, markersize=12, hue="M", add_legend=False
                )
                ts_ax.grid(axis="x", linestyle="-.")
                ts_ax.legend(
                    handles=primitive, labels=list(ts.coords["M"].values), loc="best"
                )
                ts_ax.spines["top"].set_visible(False)
                ts_ax.spines["right"].set_visible(False)
                ts_ax.spines["bottom"].set_visible(False)
                ts_ax.set_title("EOF Scores (Mode {})".format(mode + 1))
                ts_ax.set_ylabel(None)
                ts_ax.set_xlabel(None)

                map1_ax.set_title(
                    "X EOF MODE {} = {}%".format(mode + 1, ce.truncate(canvarX, 2))
                )
                map2_ax.set_title(
                    "Y EOF MODE {} = {}%".format(mode + 1, ce.truncate(canvarY, 2))
                )

                map1_ax.coastlines()
                map2_ax.coastlines()
                map1_ax.add_feature(cartopy.feature.BORDERS)
                map2_ax.add_feature(cartopy.feature.BORDERS)
                plt.show()

                # save plots
                figName = MOS + "_" + str(model) + "_EOF_mode_" + str(mode + 1) + ".png"
                fig.savefig(files_root / "figures" / figName, bbox_inches="tight")
    elif MOS == "PCR":
        for i, model in enumerate(predictor_names):
            for mode in range(nmodes):
                print(model.upper() + " - MODE {}".format(mode + 1))
                # print(model.upper() + ': EOF {}'.format(mode+1)  +' = '+ str(truncate(cancorr[0], 2)))
                fig = plt.figure(figsize=(20, 5))
                gs0 = gridspec.GridSpec(1, 3, figure=fig)
                gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])
                gs01 = gridspec.GridSpecFromSubplotSpec(5, 10, subplot_spec=gs0[1])
                gs02 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[2])
                ts = xr.concat(
                    [pxs[i].x_eof_scores.isel(Mode=mode)], "M"
                ).assign_coords({"M": ["x"]})

                map1_ax = fig.add_subplot(gs00[:, :], projection=ccrs.PlateCarree())
                ts_ax = fig.add_subplot(gs01[1:3, 1:])
                map2_ax = fig.add_subplot(gs02[:, :], projection=ccrs.PlateCarree())

                (
                    pxs[i]
                    .x_eof_loadings.isel(Mode=mode)
                    .where(pxs[i].x_eof_loadings.isel(Mode=mode) > missing_value_flag)
                    .plot(ax=map1_ax, cmap=cmap)
                )

                primitive = ts.plot.line(
                    marker="x", ax=ts_ax, markersize=12, hue="M", add_legend=False
                )
                ts_ax.grid(axis="x", linestyle="-.")
                ts_ax.legend(
                    handles=primitive, labels=list(ts.coords["M"].values), loc="best"
                )
                ts_ax.spines["top"].set_visible(False)
                ts_ax.spines["right"].set_visible(False)
                ts_ax.spines["bottom"].set_visible(False)
                ts_ax.set_title("EOF Scores (Mode {})".format(mode + 1))
                ts_ax.set_ylabel(None)
                ts_ax.set_xlabel(None)

                map1_ax.set_title("X EOF MODE {}".format(mode + 1))
                # map2_ax.set_title('Y EOF MODE {}'.format(mode+1))

                map1_ax.coastlines()
                map1_ax.add_feature(cartopy.feature.BORDERS)
                # map2_ax.coastlines()
                plt.show()

                # save plots
                figName = MOS + "_" + str(model) + "_EOF_mode_" + str(mode + 1) + ".png"
                fig.savefig(files_root / "figures" / figName, bbox_inches="tight")
    else:
        print("You will need to set MOS=CCA in order to see CCA Modes")


def plot_forecasts(
    cpt_args,
    predictand_name,
    fcsts,
    graph_orientation,
    files_root,
    predictor_names,
    MOS,
):
    prob_missing_value_flag = -1
    my_dpi = 100

    ForTitle, vmin, vmax, barcolor = ce.prepare_canvas(
        cpt_args["tailoring"], predictand_name
    )
    cmapB, cmapN, cmapA = ce.prepare_canvas(None, predictand_name, "probabilistic")

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    iidx = 1
    for i in range(len(fcsts)):
        if graph_orientation == "horizontal":
            fig = plt.figure(figsize=(18, 10), facecolor="w", dpi=my_dpi)
        else:
            fig = plt.figure(figsize=(15, 12), facecolor="w", dpi=my_dpi)

        matplotlibInstance, cartopyInstance = ce.view_probabilistic(
            fcsts[i]
            .probabilistic.where(fcsts[i].probabilistic > prob_missing_value_flag)
            .rename({"C": "M"})
            .isel(T=-1)
            / 100,
            cmap_an=cmapA,
            cmap_bn=cmapB,
            cmap_nn=cmapN,
            orientation=graph_orientation,
        )
        cartopyInstance.add_feature(cartopy.feature.BORDERS, edgecolor="black")
        cartopyInstance.set_title("")
        # cartopyInstance.axis("off")
        allaxes = matplotlibInstance.get_axes()

        cartopyInstance.spines["left"].set_color("blue")

        matplotlibInstance.savefig(
            files_root / "figures" / "Test.png",
            bbox_inches="tight",
        )  # ,pad_inches = 0)

        matplotlibInstance.clf()
        cartopyInstance.cla()

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_axis_off()
        ax1.set_title(
            predictor_names[i].upper() + " - Probabilistic Forecasts " + ForTitle
        )
        pil_img = Image.open(
            files_root / "figures" / "Test.png"
        )
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(pil_img)

        iidx = iidx + 1

        datart = (
            fcsts[i]
            .deterministic.where(fcsts[i].deterministic > missing_value_flag)
            .isel(T=-1)
        )
        if (
            any(x in predictand_name for x in ["TMAX", "TMIN", "TMEAN", "TMED"])
            and i == 0
        ):
            vmin = round(float(datart.min()) - 0.5 * 2) / 2

        art = datart.plot(
            figsize=(12, 10),
            aspect="equal",
            yincrease=True,
            subplot_kws={"projection": ccrs.PlateCarree()},
            # cbar_kwargs={'location': 'bottom',
            # "label": "Temperature (°C)"
            #'xticklabels':{'fontsize':100},
            #            },
            extend="neither",
            add_colorbar=False,
            transform=ccrs.PlateCarree(),
            cmap=barcolor,
            vmin=vmin,
            vmax=vmax,
        )

        plt.title("")
        # plt.axis("off")
        art.axes.coastlines()

        cb = plt.colorbar(art, orientation=graph_orientation)  # location='bottom')
        cb.set_label(
            label=datart.attrs["field"] + " [" + datart.attrs["units"] + "]", size=16
        )
        cb.ax.tick_params(labelsize=15)

        art.axes.add_feature(
            cartopy.feature.BORDERS, edgecolor="black"
        )  # ,linewidth=4.5
        art.axes.coastlines(edgecolor="black")  # ,linewidth=4.5
        plt.savefig(
            files_root / "figures" / "Test.png",
            bbox_inches="tight",
            pad_inches=0,
        )

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_axis_off()

        ax2.set_title(
            predictor_names[i].upper() + " - Deterministic Forecasts " + ForTitle
        )
        pil_img = Image.open(
            files_root / "figures" / "Test.png"
        )
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(pil_img)  # , aspect=4 1.45 , extent=[0, 1.45, 1.5, 0],

        iidx = iidx + 1

        # save plots
        figName = (
            MOS
            + ForTitle.replace(" ", "_")
            + "["
            + ",".join(predictor_names)
            + "]"
            + "[determinstic-probabilistic]-Forecast"
            + ".png"
        )
        fig.savefig(
            files_root / "figures" / figName,
            bbox_inches="tight",
        )
        plt.close()


def plot_mme_skill(
    predictor_names, nextgen_skill, graph_orientation, MOS, files_root
):
    # skill_metrics = ['pearson', 'spearman', 'generalized_roc', 'rank_probability_skill_score']
    # probabilistic metrics: 'generalized_roc', 'rank_probability_skill_score', 'ignorance'
    # deterministic metrics: 16! incl 3 flavors of afc!
    ##cmaps = [plt.get_cmap('RdBu', 11), plt.get_cmap('RdBu', 11), plt.get_cmap('autumn_r', 10), plt.get_cmap('autumn_r', 11) ]
    ##limits = [(-1, 1), (-1, 1), (50, 100), (0, 50)]
    # cmaps = [plt.get_cmap('cpt.correlation', 11), plt.get_cmap('cpt.correlation', 11), plt.get_cmap('RdBu_r', 10), plt.get_cmap('cpt.correlation', 11) ]
    # limits = [(-1, 1), (-1, 1), (0, 100), (-50, 50)]

    # my choices
    skill_metrics = [
        "spearman",
        "2afc",
        "generalized_roc",
        "rank_probability_skill_score",
    ]
    cmaps = [
        ce.cmaps["cpt_correlation"],
        ce.cmaps["pycpt_roc"],
        ce.cmaps["pycpt_roc"],
        ce.cmaps["cpt_correlation"],
    ]
    limits = [(-1, 1), (0, 100), (0, 100), (-50, 50)]

    cmaps[2].set_under("lightgray")
    cmaps[3].set_under("lightgray")

    ## Do not modify below
    # fig, ax = plt.subplots(nrows=len(skill_metrics), ncols=1, subplot_kw={'projection':ccrs.PlateCarree()}, figsize=(4, 5*len(skill_metrics)))
    # for j, skill_metric in enumerate(skill_metrics):
    #     ax[j].set_title(skill_metric)
    #     getattr(nextgen_skill, skill_metric).where(getattr(nextgen_skill, skill_metric) > missing_value_flag).plot(ax=ax[j], cmap=cmaps[j], vmin=limits[j][0], vmax=limits[j][1])
    #     ax[j].coastlines()
    #     ax[j].add_feature(cartopyFeature.BORDERS)

    # my plotting (taken from individual models, here with 1 row)
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(skill_metrics),
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(5 * len(skill_metrics), 1 * len(predictor_names)),
    )
    if len(predictor_names) == 1:
        ax = [ax]

    for i in [1]:
        for j, skill_metric in enumerate(skill_metrics):
            ax[j].set_title(skill_metric)
            # n = getattr(skill[i], skill_metric).where(getattr(skill[i], skill_metric) > missing_value_flag).plot(ax=ax[i][j], cmap=cmaps[j], vmin=limits[j][0], vmax=limits[j][1])
            n = (
                getattr(nextgen_skill, skill_metric)
                .where(getattr(nextgen_skill, skill_metric) > missing_value_flag)
                .plot(
                    ax=ax[j],
                    cmap=cmaps[j],
                    vmin=limits[j][0],
                    vmax=limits[j][1],
                    add_colorbar=False,
                )
            )

            ax[j].coastlines()
            ax[j].add_feature(cartopy.feature.BORDERS)
            ax[j].set_title(skill_metric.upper())

            cb = plt.colorbar(n, orientation=graph_orientation)  # location='bottom')
            cb.set_label(label=skill_metric, size=15)
            cb.ax.tick_params(labelsize=12)

    #    ax[i][0].text(-0.07, 0.55, model.upper(), va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform=ax[i][0].transAxes)

    # save plots
    figName = MOS + "_ensemble_forecast_skillMatrices.png"
    fig.savefig(
        files_root / "figures" / figName,
        bbox_inches="tight",
    )


def plot_mme_forecasts(
    graph_orientation,
    cpt_args,
    predictand_name,
    pr_fcst,
    MOS,
    files_root,
    det_fcst,
):
    missing_value_flag = -999
    prob_missing_value_flag = -1

    my_dpi = 80
    # fig = plt.figure( figsize=(9*len(fcsts), 5*len(fcsts)), dpi=my_dpi)
    # fig = plt.figure( figsize=(18, 10), dpi=my_dpi)
    if graph_orientation == "horizontal":
        fig = plt.figure(figsize=(18, 10), dpi=my_dpi)
    else:
        fig = plt.figure(figsize=(15, 12), dpi=my_dpi)

    ForTitle, vmin, vmax, barcolor = ce.prepare_canvas(
        cpt_args["tailoring"], predictand_name
    )
    cmapB, cmapN, cmapA = ce.prepare_canvas(None, predictand_name, "probabilistic")

    matplotlibInstance, cartopyInstance = ce.view_probabilistic(
        pr_fcst.where(pr_fcst > prob_missing_value_flag).rename({"C": "M"}).isel(T=-1)
        / 100,
        cmap_an=cmapA,
        cmap_bn=cmapB,
        cmap_nn=cmapN,
        orientation=graph_orientation,
    )
    cartopyInstance.add_feature(cartopy.feature.BORDERS)
    cartopyInstance.set_title("")
    # cartopyInstance.axis("off")

    figName = MOS + "_ensemble_probabilistic-deterministicForecast.png"
    plt.savefig(
        files_root / "figures" / "Test.png",
        bbox_inches="tight",
    )

    matplotlibInstance.clf()
    cartopyInstance.cla()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_axis_off()
    ax1.set_title(MOS + "_ensemble" + " - Probabilistic Forecasts " + ForTitle)
    pil_img = Image.open(
        files_root / "figures" / "Test.png"
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(pil_img)

    datart = det_fcst.where(det_fcst > missing_value_flag).isel(T=-1)
    if any(x in predictand_name for x in ["TMAX", "TMIN", "TMEAN", "TMED"]) and i == 0:
        vmin = round(float(datart.min()) - 0.5 * 2) / 2

    art = datart.plot(
        figsize=(12, 10),
        aspect="equal",
        yincrease=True,
        # size=45,
        subplot_kws={"projection": ccrs.PlateCarree()},
        extend="neither",
        add_colorbar=False,
        transform=ccrs.PlateCarree(),
        cmap=barcolor,
        vmin=vmin,
        vmax=vmax,
    )

    plt.title("")
    art.axes.coastlines()

    cb = plt.colorbar(art, orientation=graph_orientation)
    cb.set_label(label=datart.name, size=16)
    cb.ax.tick_params(labelsize=15)

    art.axes.add_feature(cartopy.feature.BORDERS, edgecolor="black")  # ,linewidth=4.5
    art.axes.coastlines(edgecolor="black")

    plt.savefig(
        files_root / "figures" / "Test.png",
        bbox_inches="tight",
    )

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_axis_off()
    ax2.set_title(MOS + "_ensemble" + " - Deterministic Forecasts " + ForTitle)
    pil_img = Image.open(
        files_root / "figures" / "Test.png"
    )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(pil_img)

    fig.savefig(
        files_root / "figures" / figName,
        bbox_inches="tight",
    )
    plt.close()


def plot_mme_flex_forecasts(
    predictand_name,
    graph_orientation,
    exceedance_prob,
    point_latitude,
    point_longitude,
    download_args,
    threshold,
    fcst_scale,
    climo_scale,
    fcst_mu,
    climo_mu,
    Y2,
    is_transformed,
    ntrain,
    Y,
    MOS,
    files_root,
):
    if point_latitude < float(
        download_args["predictand_extent"]["south"]
    ) or point_latitude > float(download_args["predictand_extent"]["north"]):
        point_latitude = round(
            (
                download_args["predictand_extent"]["south"]
                + download_args["predictand_extent"]["north"]
            )
            / 2,
            2,
        )

    if point_longitude < float(
        download_args["predictand_extent"]["west"]
    ) or point_longitude > float(download_args["predictand_extent"]["east"]):
        point_longitude = round(
            (
                download_args["predictand_extent"]["west"]
                + download_args["predictand_extent"]["east"]
            )
            / 2,
            2,
        )

    # plot exceedance probability map

    ForTitle, vmin, vmax, mark, barcolor = ce.prepare_canvas("POE", predictand_name)

    # setting up canvas on which to draw

    if graph_orientation == "horizontal":
        fig = plt.figure(figsize=(15, 10))
    else:
        fig = plt.figure(figsize=(10, 20))

    # fig = plt.figure()
    gs0 = gridspec.GridSpec(4, 1, figure=fig)
    gs00 = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=gs0[:3])
    gs11 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[3])
    gs01 = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=gs11[0])
    gs02 = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=gs11[1])

    map_ax = fig.add_subplot(gs00[:, :], projection=ccrs.PlateCarree(), aspect="auto")
    cdf_ax = fig.add_subplot(gs01[:, :], aspect="auto")
    pdf_ax = fig.add_subplot(gs02[:, :], aspect="auto")

    # plot the map
    art = exceedance_prob.transpose("Y", "X", ...).plot(
        cmap=barcolor, ax=map_ax, vmin=vmin, vmax=vmax
    )
    map_ax.scatter(
        [point_longitude],
        [point_latitude],
        marker="x",
        s=100,
        color=mark,
        transform=ccrs.PlateCarree(),
    )
    coasts = art.axes.coastlines()
    art.axes.add_feature(cartopy.feature.BORDERS)
    title = map_ax.set_title("(a) Probabilities of Exceedance")

    # point calculations - select the nearest point to the lat/lon the user wanted to plot curves
    point_threshold = float(
        threshold.sel(
            **{"X": point_longitude, "Y": point_latitude}, method="nearest"
        ).values
    )
    point_fcst_scale = float(
        fcst_scale.sel(
            **{"X": point_longitude, "Y": point_latitude}, method="nearest"
        ).values
    )
    point_climo_scale = float(
        climo_scale.sel(
            **{"X": point_longitude, "Y": point_latitude}, method="nearest"
        ).values
    )
    point_fcst_mu = float(
        fcst_mu.sel(
            **{"X": point_longitude, "Y": point_latitude}, method="nearest"
        ).values
    )
    point_climo_mu = float(
        climo_mu.sel(
            **{"X": point_longitude, "Y": point_latitude}, method="nearest"
        ).values
    )
    point_climo = np.squeeze(
        Y2.sel(**{"X": point_longitude, "Y": point_latitude}, method="nearest").values
    )
    point_climo.sort()

    if is_transformed:
        point_climo_mu_nontransformed = float(
            Y.mean("T")
            .sel(**{"X": point_longitude, "Y": point_latitude}, method="nearest")
            .values
        )
        point_climo_std_nontransformed = float(
            Y.std("T")
            .sel(**{"X": point_longitude, "Y": point_latitude}, method="nearest")
            .values
        )

    x = point_climo
    x1 = np.linspace(x.min(), x.max(), 1000)
    cprobth = (
        sum(x >= point_threshold) / x.shape[0]
    )  # round(t.sf(point_threshold, ntrain, loc=point_climo_mu, scale=point_climo_scale),2)
    fprobth = round(
        t.sf(point_threshold, ntrain, loc=point_fcst_mu, scale=point_fcst_scale), 2
    )

    # POE plot
    cdf_ax.plot(
        x,
        [sum(x >= x[i]) / x.shape[0] for i in range(x.shape[0])],
        "g-",
        lw=2,
        marker="x",
        alpha=0.8,
        label="clim (empirical)",
    )
    cdf_ax.plot(
        x1,
        t.sf(x1, ntrain, loc=point_fcst_mu, scale=point_fcst_scale),
        "r-",
        lw=1,
        alpha=0.8,
        label="fcst",
    )
    cdf_ax.plot(
        x1,
        norm.sf(x1, loc=point_climo_mu, scale=point_fcst_scale),
        "b-",
        lw=1,
        alpha=0.8,
        label="clim (fitted)",
    )

    cdf_ax.plot(point_threshold, fprobth, "ok")
    cdf_ax.plot(point_threshold, cprobth, "ok")
    cdf_ax.axvline(x=point_threshold, color="k", linestyle="--")
    cdf_ax.set_title(" (b) Point Probabilities of Exceedance")
    cdf_ax.set_xlabel(Y.name.upper())
    cdf_ax.set_ylabel("Probability (%)")
    cdf_ax.legend(loc="best", frameon=False)

    # PDF plot
    # fpdf=t.pdf(x1, ntrain, loc=point_fcst_mu, scale=np.sqrt(point_fcst_scale))
    fpdf = t.pdf(x1, ntrain, loc=point_fcst_mu, scale=point_fcst_scale)

    pdf_ax.plot(
        x1,
        norm.pdf(x1, loc=point_climo_mu, scale=point_climo_scale),
        "b-",
        alpha=0.8,
        label="clim (fitted)",
    )  # clim pdf in blue
    pdf_ax.plot(x1, fpdf, "r-", alpha=0.8, label="fcst")  # fcst PDF in red
    pdf_ax.hist(
        point_climo, density=True, histtype="step", label="clim (empirical)"
    )  # want this in GREEN

    pdf_ax.axvline(x=point_threshold, color="k", linestyle="--")
    pdf_ax.legend(loc="best", frameon=False)
    pdf_ax.set_title("(c) Point Probability Density Functions")
    pdf_ax.set_xlabel(Y.name.upper())
    pdf_ax.set_ylabel("")

    if is_transformed:
        newticks = [-2, -1, 0, 1, 2]
        pdf_ax.set_xticks(
            newticks,
            [
                round(
                    i * point_climo_std_nontransformed + point_climo_mu_nontransformed,
                    2,
                )
                for i in newticks
            ],
            rotation=0,
        )
        cdf_ax.set_xticks(
            newticks,
            [
                round(
                    i * point_climo_std_nontransformed + point_climo_mu_nontransformed,
                    2,
                )
                for i in newticks
            ],
            rotation=0,
        )

    # save plot
    figName = MOS + "_flexForecast_probExceedence.png"
    plt.savefig(
        files_root / "figures" / figName,
        bbox_inches="tight",
    )
