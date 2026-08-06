import type { Component } from "@bigai-nlco/bcg-tui";
import { ExtensionSelectorComponent } from "./components/extension-selector.ts";
import { OAuthSelectorComponent, type AuthSelectorProvider } from "./components/oauth-selector.ts";

/**
 * Auth selector flows (step 11.3): the login/logout selector orchestration
 * extracted from InteractiveMode. UI primitives (showSelector) and backend
 * operations (provider login/logout) are injected; the flows decide what to
 * show and how callbacks dispatch.
 */
export interface AuthSelectorsDeps {
	showStatus(message: string): void;
	showError(message: string): void;
	requestRender(): void;
	showSelector(create: (done: () => void) => { component: Component; focus: Component }): void;
	getLoginProviderOptions(authType?: "oauth" | "api_key"): AuthSelectorProvider[];
	getLogoutProviderOptions(): Promise<AuthSelectorProvider[]>;
	startProviderLogin(option: AuthSelectorProvider): Promise<void> | void;
	/** Backend logout plus provider-count refresh; the flow owns the messages. */
	logout(providerId: string): Promise<void>;
}

export function showLoginAuthTypeSelector(
	deps: AuthSelectorsDeps,
	providerOptions?: AuthSelectorProvider[],
): void {
	const oauthProvider = providerOptions?.find((provider) => provider.authType === "oauth");
	const oauthLoginLabel =
		oauthProvider?.method && "loginLabel" in oauthProvider.method
			? oauthProvider.method.loginLabel
			: undefined;
	const subscriptionLabel = oauthLoginLabel ?? "Sign in with an account";
	const apiKeyLabel = "Sign in with an API key";
	const availableAuthTypes = providerOptions
		? new Set(providerOptions.map((provider) => provider.authType))
		: new Set<AuthSelectorProvider["authType"]>(["oauth", "api_key"]);
	const options: string[] = [];
	if (availableAuthTypes.has("oauth")) {
		options.push(subscriptionLabel);
	}
	if (availableAuthTypes.has("api_key")) {
		options.push(apiKeyLabel);
	}

	if (options.length === 0) {
		deps.showStatus("No login methods available.");
		return;
	}

	if (providerOptions && options.length === 1) {
		const providerOption = providerOptions[0];
		if (providerOption) {
			void deps.startProviderLogin(providerOption);
		}
		return;
	}

	const title = providerOptions?.[0]
		? `Select authentication method for ${providerOptions[0].name}:`
		: "Select authentication method:";
	deps.showSelector((done) => {
		const selector = new ExtensionSelectorComponent(
			title,
			options,
			(option) => {
				done();
				const authType = option === subscriptionLabel ? "oauth" : "api_key";
				if (providerOptions) {
					const providerOption = providerOptions.find((provider) => provider.authType === authType);
					if (providerOption) {
						void deps.startProviderLogin(providerOption);
					}
					return;
				}
				showLoginProviderSelector(deps, authType);
			},
			() => {
				done();
				deps.requestRender();
			},
		);
		return { component: selector, focus: selector };
	});
}

export function showLoginProviderSelector(
	deps: AuthSelectorsDeps,
	authType?: AuthSelectorProvider["authType"],
	initialSearchInput?: string,
): void {
	const providerOptions = deps.getLoginProviderOptions(authType);
	if (providerOptions.length === 0) {
		const message =
			authType === "oauth"
				? "No subscription providers available."
				: authType === "api_key"
					? "No API key providers available."
					: "No login providers available.";
		deps.showStatus(message);
		return;
	}

	deps.showSelector((done) => {
		const selector = new OAuthSelectorComponent(
			"login",
			providerOptions,
			async (providerId, selectedAuthType) => {
				done();

				const providerOption = providerOptions.find(
					(provider) => provider.id === providerId && provider.authType === selectedAuthType,
				);
				if (!providerOption) {
					return;
				}

				await deps.startProviderLogin(providerOption);
			},
			() => {
				done();
				if (authType) {
					showLoginAuthTypeSelector(deps);
				} else {
					deps.requestRender();
				}
			},
			initialSearchInput,
		);
		return { component: selector, focus: selector };
	});
}

export async function showOAuthSelector(deps: AuthSelectorsDeps, mode: "login" | "logout"): Promise<void> {
	if (mode === "login") {
		showLoginAuthTypeSelector(deps);
		return;
	}

	const providerOptions = await deps.getLogoutProviderOptions();
	if (providerOptions.length === 0) {
		deps.showStatus(
			"No stored credentials to remove. /logout only removes credentials saved by /login; environment variables and models.json config are unchanged.",
		);
		return;
	}

	deps.showSelector((done) => {
		const selector = new OAuthSelectorComponent(
			mode,
			providerOptions,
			async (providerId: string) => {
				done();

				const providerOption = providerOptions.find((provider) => provider.id === providerId);
				if (!providerOption) {
					return;
				}

				try {
					await deps.logout(providerOption.id);
					const message =
						providerOption.authType === "oauth"
							? `Logged out of ${providerOption.name}`
							: `Removed stored API key for ${providerOption.name}. Environment variables and models.json config are unchanged.`;
					deps.showStatus(message);
				} catch (error: unknown) {
					deps.showError(`Logout failed: ${error instanceof Error ? error.message : String(error)}`);
				}
			},
			() => {
				done();
				deps.requestRender();
			},
		);
		return { component: selector, focus: selector };
	});
}
