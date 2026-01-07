import Head from 'next/head';

export default function App({ Component, pageProps }) {
  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <style jsx global>{`
        html, body {
          margin: 0;
          padding: 0;
          background: #212121;
          overflow: hidden;
        }
        * {
          box-sizing: border-box;
        }
      `}</style>
      <Component {...pageProps} />
    </>
  );
}
