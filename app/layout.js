import { Inter } from "next/font/google";
import "./globals.css";
import Sidebar from "./ui/Sidebar";
import Provider from "./ui/Provider";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function RootLayout({ children, session }) {
  return (
    <html lang="en">
      <body className={`lg:flex lg:flex-row ${inter.className}`}>
        <Provider session={session}>
          <Sidebar/>
          {children}
        </Provider>
      </body>
    </html>
  );
}
