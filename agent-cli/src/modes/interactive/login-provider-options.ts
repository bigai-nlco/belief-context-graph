import type { AuthSelectorProvider } from "./components/oauth-selector.ts";

/**
 * Login provider option building (step 11.3): extracted from InteractiveMode.
 * Pure option assembly over a minimal model-runtime auth view; the session
 * supplies the runtime, the UI flow stays in InteractiveMode.
 */
export interface ModelRuntimeAuthView {
	getProviders(): ReadonlyArray<{
		id: string;
		name: string;
		auth: { oauth?: unknown; apiKey?: unknown };
	}>;
	getProviderAuthStatus(providerId: string): {
		configured: boolean;
		label?: string;
		source?: string;
	};
	isUsingOAuth(providerId: string): boolean;
	getProvider(providerId: string): { id: string; name: string } | undefined;
	listCredentials(): Promise<ReadonlyArray<{ providerId: string; type: "oauth" | "api_key" }>>;
}

export function buildLoginProviderOptions(
	runtime: ModelRuntimeAuthView,
	authType?: "oauth" | "api_key",
): AuthSelectorProvider[] {
	const options: AuthSelectorProvider[] = [];
	for (const provider of runtime.getProviders()) {
		const authStatus = runtime.getProviderAuthStatus(provider.id);
		const status = authStatus.configured
			? {
					type: runtime.isUsingOAuth(provider.id) ? ("oauth" as const) : ("api_key" as const),
					source: authStatus.label ?? authStatus.source,
				}
			: undefined;
		if ((!authType || authType === "oauth") && provider.auth.oauth) {
			options.push({
				id: provider.id,
				name: provider.name,
				authType: "oauth",
				method: provider.auth.oauth as never,
				status,
			});
		}
		if ((!authType || authType === "api_key") && provider.auth.apiKey) {
			options.push({
				id: provider.id,
				name: provider.name,
				authType: "api_key",
				method: provider.auth.apiKey as never,
				status,
			});
		}
	}
	return options.sort((a, b) => a.name.localeCompare(b.name));
}

export async function buildLogoutProviderOptions(
	runtime: ModelRuntimeAuthView,
): Promise<AuthSelectorProvider[]> {
	return (await runtime.listCredentials())
		.map(({ providerId, type }) => ({
			id: providerId,
			name: runtime.getProvider(providerId)?.name ?? providerId,
			authType: type,
			status: { type, source: "stored credential" },
		}))
		.sort((a, b) => a.name.localeCompare(b.name));
}

export function findLoginProviderOptions(
	runtime: ModelRuntimeAuthView,
	providerRef: string,
): AuthSelectorProvider[] {
	const normalizedProviderRef = providerRef.trim().toLowerCase();
	if (!normalizedProviderRef) {
		return [];
	}

	return buildLoginProviderOptions(runtime).filter(
		(provider) =>
			provider.id.toLowerCase() === normalizedProviderRef ||
			provider.name.toLowerCase() === normalizedProviderRef,
	);
}
