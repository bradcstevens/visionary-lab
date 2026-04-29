import React, { Suspense } from "react";
import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider"
import { AppSidebar } from "@/components/app-sidebar";
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { VideoQueueProvider } from "@/context/video-queue-context";
import { JobsProvider } from "@/context/jobs-context";
import { ImageSettingsProvider } from "@/context/image-settings-context";
import { FolderProvider } from "@/context/folder-context";
import { ActivityLogProvider } from "@/context/activity-log-context";
import { ActivityLogToggle } from "@/components/activity-log/ActivityLogToggle";
import { ActivityLogPanel } from "@/components/activity-log/ActivityLogPanel";
import { VideoQueueClient } from "@/components/video-queue-client";
import { RefreshJobsButton } from "@/components/refresh-jobs-button";
import { Toaster } from "@/components/ui/sonner";
import { AnimatedLayout } from "@/components/animated-layout";
import Script from "next/script";

type RootLayoutProps = {
  children: React.ReactNode
}

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Visionary Lab",
  description: "AI-powered Content Generation",
  manifest: "/manifest.json",
  icons: {
    apple: "/logo/icon-192.png",
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "Visionary Lab",
  },
  other: {
    "mobile-web-app-capable": "yes",
    "apple-mobile-web-app-capable": "yes",
    "apple-mobile-web-app-status-bar-style": "default",
    "msapplication-TileColor": "#000000",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  themeColor: "#000000",
};

export default async function RootLayout({ children }: RootLayoutProps) {

  return (
    <html lang="en" suppressHydrationWarning className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link rel="dns-prefetch" href={process.env.NEXT_PUBLIC_API_HOSTNAME || "localhost"} />
        <meta name="format-detection" content="telephone=no" />
      </head>
      <body className="overflow-hidden">
          <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
          >
            <VideoQueueProvider>
                <JobsProvider>
                  <ImageSettingsProvider>
                    <FolderProvider>
                    <ActivityLogProvider>
                    {/* Main layout with sidebar */}
                    <div className="relative flex min-h-screen h-screen">              
                      {/* Content area with sidebar */}
                      <SidebarProvider
                        style={
                          {
                            "--sidebar-width": "12rem",
                          } as React.CSSProperties
                        }
                        className="flex h-full w-full"
                      >
                        {/* Sidebar for navigation - wrapped in Suspense to fix hydration errors */}
                        <Suspense fallback={
                          <div className="w-[var(--sidebar-width)] shrink-0 border-r h-full" />
                        }>
                          <AppSidebar />
                        </Suspense>
                        <SidebarInset className="flex-1 flex flex-col h-full w-full">
                          <div className="flex h-14 items-center gap-2 border-b shrink-0 px-3">
                            <SidebarTrigger />
                            <Separator orientation="vertical" className="mx-2 h-4" />
                            <div className="ml-auto flex items-center space-x-2">
                              <ActivityLogToggle />
                              <RefreshJobsButton />
                              <VideoQueueClient />
                            </div>
                          </div>
                          <div className="flex flex-1 min-h-0">
                            <div className="flex-1 overflow-auto transition-all duration-200">
                              <AnimatedLayout>
                                {children}
                              </AnimatedLayout>
                            </div>
                            <ActivityLogPanel />
                          </div>
                        </SidebarInset>
                      </SidebarProvider>
                    </div>
                    <Toaster />
                    </ActivityLogProvider>
                    </FolderProvider>
                  </ImageSettingsProvider>
                </JobsProvider>
              </VideoQueueProvider>
          </ThemeProvider>

        
        {/* Service Worker Registration */}
        <Script
          id="sw-register"
          strategy="afterInteractive"
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                ${process.env.NODE_ENV === 'production' ? `
                window.addEventListener('load', function() {
                  navigator.serviceWorker.register('/sw.js')
                    .then(function(registration) {
                      console.log('SW registered: ', registration);
                    })
                    .catch(function(registrationError) {
                      console.log('SW registration failed: ', registrationError);
                    });
                });
                ` : `
                // Development: unregister any service worker from prior production builds
                // and clear all caches to prevent stale chunk intercepts (404s on /_next/...).
                navigator.serviceWorker.getRegistrations().then(function(registrations) {
                  registrations.forEach(function(r) { r.unregister(); });
                });
                if (typeof caches !== 'undefined') {
                  caches.keys().then(function(keys) {
                    keys.forEach(function(k) { caches.delete(k); });
                  });
                }
                `}
              }
            `,
          }}
        />
        
        {/* Preload critical resources */}
        <Script
          id="resource-preload"
          strategy="afterInteractive"
          dangerouslySetInnerHTML={{
            __html: `
              // Preload critical API endpoints
              if ('fetch' in window) {
                fetch('/api/environment', { method: 'HEAD' }).catch(() => {});
              }
              
              // Prefetch likely navigation targets
              if ('requestIdleCallback' in window) {
                requestIdleCallback(() => {
                  const links = ['/gallery', '/new-image', '/new-video'];
                  links.forEach(link => {
                    const linkEl = document.createElement('link');
                    linkEl.rel = 'prefetch';
                    linkEl.href = link;
                    document.head.appendChild(linkEl);
                  });
                });
              }
            `,
          }}
        />
      </body>
    </html>
  )
}
