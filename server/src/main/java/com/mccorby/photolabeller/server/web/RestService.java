package com.mccorby.photolabeller.server.web;

import com.google.common.io.Files;
import com.google.common.util.concurrent.RateLimiter;
import com.mccorby.photolabeller.server.BasicRoundController;
import com.mccorby.photolabeller.server.FederatedServerImpl;
import com.mccorby.photolabeller.server.core.FederatedAveragingStrategy;
import com.mccorby.photolabeller.server.core.datasource.*;
import com.mccorby.photolabeller.server.core.domain.model.*;
import com.mccorby.photolabeller.server.core.domain.repository.ServerRepository;
import org.apache.commons.io.IOUtils;
import org.glassfish.jersey.media.multipart.FormDataParam;
import util.Hasher;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

@Path("/service/federatedservice")
public class RestService {

    private static FederatedServer federatedServer;

    /**
     * initialize classes and file data source when server start
     */
    public RestService() throws IOException {
        if (federatedServer == null) {
            // TODO Inject!
            // TODO Properties to SharedConfig
            Properties properties = new Properties();
            properties.load(new FileInputStream("./server/local.properties"));

            java.nio.file.Path rootPath = Paths.get(properties.getProperty("model_dir"));
            FileDataSource fileDataSource = new FileDataSourceImpl(rootPath);
            MemoryDataSource memoryDataSource = new MemoryDataSourceImpl();
            ServerRepository repository = new ServerRepositoryImpl(fileDataSource, memoryDataSource);
            Logger logger = System.out::println;
            UpdatesStrategy updatesStrategy = new FederatedAveragingStrategy(repository, Integer.valueOf(properties.getProperty("layer_index")));

            UpdatingRound currentUpdatingRound = repository.retrieveCurrentUpdatingRound();

            long timeWindow = Long.valueOf(properties.getProperty("time_window"));
            int minUpdates = Integer.valueOf(properties.getProperty("min_updates"));

            RoundController roundController = new BasicRoundController(repository, currentUpdatingRound, timeWindow, minUpdates);

            federatedServer = FederatedServerImpl.Companion.getInstance();
            federatedServer.initialise(repository, updatesStrategy, roundController, logger, properties);

            // We're starting a new round when the server starts
            roundController.startRound();
        }
    }

    /**
     * server health check
     */
    @GET
    @Path("/available")
    @Produces(MediaType.TEXT_PLAIN)
    public String available() {

        GuavaRateLimiter limiter = new GuavaRateLimiter();
        limiter.initLimiter();

        System.out.println("[available]");
        return "yes";
    }

    /**
     * client POST on device trained model gradients
     * @param is model gradients
     * @param samples the number of image size to train this model gradients
     * @return [Boolean] update efficient or not
     */
    @POST
    @Consumes(MediaType.MULTIPART_FORM_DATA)
    @Path("/model")
    public Boolean pushGradient(@FormDataParam("file") InputStream is, @FormDataParam("samples") int samples) throws IOException {
        System.out.println("[pushGradient] samples: " + samples);
        if (is == null) {
            return false;
        } else {
            byte[] bytes = IOUtils.toByteArray(is);

            /**
             * compare update model param hashes to redis
             */
            File targetFile = new File("src/main/resources/targetFile.tmp");
            Files.write(bytes, targetFile);
            Hasher hasher = new Hasher();
            String modelParamsHash = hasher.hash(targetFile);
            // TODO compare hash to redis, if not exists, push to redis [hash: file]


            federatedServer.pushUpdate(bytes, samples);
            return true;
        }
    }

    /**
     * client get model if client side have no embedded model exits when app starts
     *
     * @return model data file
     */
    @GET
    @Path("/model")
    @Produces(MediaType.APPLICATION_OCTET_STREAM)
    public Response getFile() {
        System.out.println("[getFile]");

        GuavaRateLimiter limiter = new GuavaRateLimiter();
        limiter.initLimiter();

        File file = federatedServer.getModelFile();
        String fileName = federatedServer.getUpdatingRound().getModelVersion() + ".zip";
        Response.ResponseBuilder response = Response.ok(file);
        response.header("Content-Disposition", "attachment; filename=\"" + fileName + "\"");
        return response.build();
    }

    /**
     * get the current training round data, from currentRound.json
     * @return [String] current round
     */
    @GET
    @Path("/currentRound")
    @Produces(MediaType.APPLICATION_JSON)
    public String getCurrentRound() {
        System.out.print("[getCurrentRound()]");

        GuavaRateLimiter limiter = new GuavaRateLimiter();
        limiter.initLimiter();

        return federatedServer.getUpdatingRoundAsJson();
    }
}

/**
 * Guava rate limiter
 */
class GuavaRateLimiter {
    public void initLimiter() {
        RateLimiter rateLimiter = RateLimiter.create(100.0);
        Boolean acquired =  rateLimiter.tryAcquire(1000L, TimeUnit.MILLISECONDS);

        if (!acquired) {
            Response.ResponseBuilder resp =  Response.ok();
            resp.status(403, "too many request, please retry later!");
        }
    }
}
