import {Construct} from "constructs";
import {SagemakerEndpointConfiguration} from "@cdktf/provider-aws/lib/sagemaker-endpoint-configuration";
import { config } from './config';
import {SagemakerEndpoint} from "@cdktf/provider-aws/lib/sagemaker-endpoint";
import {SagemakerModel} from "@cdktf/provider-aws/lib/sagemaker-model";
import {DataAwsIamPolicyDocument} from "@cdktf/provider-aws/lib/data-aws-iam-policy-document";
import {IamRole} from "@cdktf/provider-aws/lib/iam-role";
import {PocketVPC} from "@pocket-tools/terraform-modules";
import {IamPolicyAttachment} from "@cdktf/provider-aws/lib/iam-policy-attachment";


export class Sagemaker extends Construct {
  public readonly llmEndpoint: ApplicationSagemaker;
  public readonly embeddingsEndpoint: ApplicationSagemaker;

  constructor(scope: Construct, name: string, vpc: PocketVPC) {
    super(scope, name);

    this.llmEndpoint = new ApplicationSagemaker(this, `llm`, {
        // https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
        modelId: 'mistralai/Mistral-7B-Instruct-v0.1',
        // https://github.com/aws/deep-learning-containers/blob/master/huggingface/pytorch/inference/docker/2.0/py3/cu118/Dockerfile.gpu
        // https://huggingface.co/blog/sagemaker-huggingface-llm
        modelImage: '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04',
        // The model we are using requires a GPU ml.g4dn.xlarge is the cheapest instance with a gpu, but we use ml.g5.2xlarge because this model requires more memory allocation.
        instanceType: 'ml.g5.2xlarge',
        instanceCount: 1,
        vpc,
        modelName: 'llm',
        // https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
        // The sagemaker setup in the llm code expects a specific format, which is what the text-generation pipeline enables.
        // You can look at the model on HuggingFace to determine what the HF_TASK needs to be set to.
        // If not set, the image will try and determine it from the Architecture set in Config.json of your model
        // https://github.com/aws/sagemaker-huggingface-inference-toolkit/blob/main/src/sagemaker_huggingface_inference_toolkit/transformers_utils.py#L80-L92
        huggingFaceTask: 'text-generation'
    });

   this.embeddingsEndpoint = new ApplicationSagemaker(this, `embeddings`, {
        // https://huggingface.co/BAAI/bge-small-en-v1.5
        modelId: 'BAAI/bge-small-en-v1.5',
        // https://github.com/aws/deep-learning-containers/blob/master/huggingface/pytorch/inference/docker/2.0/py3/cu118/Dockerfile.gpu
        // https://huggingface.co/blog/sagemaker-huggingface-llm
        modelImage: '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04',
        // The model we are using requires a GPU ml.g4dn.xlarge is the cheapest instance with a gpu.
        instanceType: 'ml.g4dn.xlarge',
        instanceCount: 1,
        vpc,
        modelName: 'embeddings',
        // https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
        // The sagemaker setup in the embeddings code expects a list of strings, which is what the text-classification pipeline enables.
        // You can look at the model on HuggingFace to determine what the HF_TASK needs to be set to.
        // If not set, the image will try and determine it from the Architecture set in Config.json of your model
        // https://github.com/aws/sagemaker-huggingface-inference-toolkit/blob/main/src/sagemaker_huggingface_inference_toolkit/transformers_utils.py#L80-L92
        huggingFaceTask: 'text-classification'
   });
  }
}

class ApplicationSagemaker extends Construct {

    public readonly configuration: SagemakerEndpointConfiguration
    public readonly model: SagemakerModel
    public readonly endpoint: SagemakerEndpoint

  constructor(scope: Construct, name: string, options: {vpc: PocketVPC, modelId: string, modelImage: string, modelName: string, instanceType: string, instanceCount: number, huggingFaceTask: string}) {
    super(scope, name);
    this.model = this.createModel({vpc: options.vpc, modelId: options.modelId, modelImage: options.modelImage, modelName: options.modelName, huggingFaceTask: options.huggingFaceTask})
    this.configuration = this.createConfiguration({model: this.model, modelName: options.modelName, instanceType: options.instanceType, instanceCount: options.instanceCount})
    this.endpoint = this.createEndpoint({configuration: this.configuration, modelName: options.modelName})
  }


  private createModel(options: {vpc: PocketVPC, modelId: string, modelImage: string, modelName: string, huggingFaceTask: string}): SagemakerModel {
        const role = new IamRole(this, `model_role`, {
            name: `${config.shortName}-${config.environment}-${options.modelName}-ExecutionRole`,
            assumeRolePolicy: new DataAwsIamPolicyDocument(this, `model_assume_role_policy`, {
                statement: [{
                    actions: ["sts:AssumeRole"],
                    principals: [
                        {
                            type: "Service",
                            identifiers: ["sagemaker.amazonaws.com"]
                        }
                    ]
                }]
                }
            ).json ,
            managedPolicyArns: ['arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'],
            tags: config.tags
        })

        return new SagemakerModel(this, `model`, {
            name: `${config.shortName}-${config.environment}-${options.modelName}`,
            primaryContainer: {
                //https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-text-generation-inference-containers
                //https://huggingface.co/blog/sagemaker-huggingface-llm
                image: options.modelImage,
                mode: "SingleModel",
                environment: {
                    // https://github.com/aws/sagemaker-huggingface-inference-toolkit
                    "HF_MODEL_ID": options.modelId,
                    "HF_TASK": options.huggingFaceTask,
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": options.vpc.region,
                    'SM_NUM_GPUS': '1', // Number of GPU used per replica
                    // 'MAX_INPUT_LENGTH': '512',//  # Max length of input text
                }
            },
            executionRoleArn: role.arn,
            tags: config.tags,
            vpcConfig: {
                subnets: options.vpc.privateSubnetIds,
                securityGroupIds: options.vpc.internalSecurityGroups.ids
            },
            dependsOn: [role]
        })
    }

     private createConfiguration(options: { model: SagemakerModel, modelName: string, instanceType: string, instanceCount: number }): SagemakerEndpointConfiguration {
      return new SagemakerEndpointConfiguration(this, `config`, {
            name: `${config.shortName}-${config.environment}-${options.modelName}`,
            productionVariants: [{
                variantName: 'variant-1',
                modelName: options.model.name,
                // Because we are currently using HuggingFace which downloads the model on startup (and needs a startup healthcheck)
                // and because we use a VPC config we can not use Serverless.
                // In the future when we build our own image, we could probably use serverless.
                containerStartupHealthCheckTimeoutInSeconds: 500,
                initialInstanceCount: options.instanceCount,
                instanceType: options.instanceType,
            }],
            tags: config.tags,
            dependsOn: [options.model]
      })
    }

    private createEndpoint(options: {configuration: SagemakerEndpointConfiguration, modelName: string}): SagemakerEndpoint {
        return new SagemakerEndpoint(this, `endpoint`, {
            endpointConfigName: options.configuration.name,
            name: `${config.shortName}-${config.environment}-${options.modelName}`,
            deploymentConfig: {
                blueGreenUpdatePolicy: {
                    maximumExecutionTimeoutInSeconds: 600,
                    trafficRoutingConfiguration: {
                        type: "ALL_AT_ONCE",
                        waitIntervalInSeconds: 300
                    }
                }
            },
            tags: config.tags,
            dependsOn: [options.configuration]
        })
    }
}