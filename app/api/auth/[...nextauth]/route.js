import { addUser } from '@/lib/actions';
import { getUserByEmail } from '@/lib/data';
import NextAuth from 'next-auth';
import GitHubProvider from 'next-auth/providers/github'
const handler = NextAuth({
  providers: [
    GitHubProvider({
      clientId:process.env.NEXT_PUBLIC_GITHUB_CLIENT_ID,
      clientSecret:process.env.GITHUB_CLIENT_SECRET
    }),
  ],
  session:{
    jwt: true,
    maxAge: 30 * 24 * 60 * 60, 
    updateAge: 24 * 60 * 60,
  },
  callbacks :{
    async session({ session }) {
        const sessionUser = await getUserByEmail(session.user.email); 
        if (sessionUser) {
          session.user.id = sessionUser.id
        }
        return session;
    },
    async signIn({ profile }) {
        const existingUser = await getUserByEmail(profile.email);
        if (!existingUser) {
            const newUser = {
                name: profile.login,
                email: profile.email,
                image: profile.image || profile.avatar_url,
                risk: 'medium' 
              };
            await addUser(newUser);
        }
        return true;
    } 
}
}); 

export { handler as GET, handler as POST};
