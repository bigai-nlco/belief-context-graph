export function areExperimentalFeaturesEnabled(): boolean {
	return process.env.BCG_EXPERIMENTAL === "1";
}
